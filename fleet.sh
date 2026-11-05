#!/usr/bin/env bash
# Search for an LXMF address prefix on the GPUs of several machines at once.
# The first machine to find one wins: the others are stopped, the identity
# is copied here as <prefix>.identity, and every remote copy is deleted.
#
#   ./fleet.sh c0ffee1234                      hosts from ./hosts
#   ./fleet.sh -H gpu1,gpu2 c0ffee1234         these hosts
#   ./fleet.sh -n 2 c0ffee1234                 the first 2 hosts only
#
# Hosts come from -H, else $LXMF_VANITY_HOSTS, else the file ./hosts (one
# ssh destination per line; # comments allowed). Each needs ssh access
# without a password prompt, an NVIDIA GPU with CUDA, and Rust
# (rustup.rs). This script copies the source over and builds it there.
#
# Settings (environment):
#   CUDA_LIB     where libnvrtc and libcuda live  (/usr/local/cuda/lib64)
#   REMOTE_DIR   source folder on each host       (lxmf-vanity)
#   REPORT_SECONDS  progress interval             (10)
set -euo pipefail

CUDA_LIB=${CUDA_LIB:-/usr/local/cuda/lib64}
REMOTE_DIR=${REMOTE_DIR:-lxmf-vanity}
REPORT_SECONDS=${REPORT_SECONDS:-10}
here=$(cd "$(dirname "$0")" && pwd)

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

host_list="" limit=0
while getopts "H:n:h" opt; do
    case $opt in
        H) host_list=$OPTARG ;;
        n) limit=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[[ $# -eq 1 ]] || usage

if [[ -z $host_list ]]; then
    if [[ -n ${LXMF_VANITY_HOSTS:-} ]]; then
        host_list=$LXMF_VANITY_HOSTS
    elif [[ -f $here/hosts ]]; then
        host_list=$(sed 's/#.*//' "$here/hosts" | xargs)
    else
        echo "No hosts: pass -H host1,host2, set LXMF_VANITY_HOSTS, or list them in $here/hosts." >&2
        exit 1
    fi
fi
read -r -a HOSTS <<<"${host_list//,/ }"
if (( limit > 0 && limit < ${#HOSTS[@]} )); then HOSTS=("${HOSTS[@]:0:limit}"); fi
(( ${#HOSTS[@]} > 0 )) || { echo "No hosts given." >&2; exit 1; }

prefix=$(printf '%s' "$1" | tr 'A-F' 'a-f')
[[ $prefix =~ ^[0-9a-f]{1,32}$ ]] || { echo "'$1' isn't hex: use 0-9 and a-f only (zero, not the letter O)." >&2; exit 1; }
out="$prefix.identity"
[[ -e $out ]] && { echo "$out already exists; not overwriting it." >&2; exit 1; }
runs="$REMOTE_DIR-runs"
binary="$REMOTE_DIR/target/release/lxmf-vanity"
logs=$(mktemp -d)

echo "Building on ${HOSTS[*]}..."
for h in "${HOSTS[@]}"; do
    (rsync -a --exclude target --exclude hosts "$here/" "$h:$REMOTE_DIR/" &&
        ssh "$h" "cd $REMOTE_DIR && ~/.cargo/bin/cargo build -q --release --features cuda") >"$logs/$h.build" 2>&1 &
done
if ! wait; then
    for f in "$logs"/*.build; do [[ -s $f ]] && { echo "== $(basename "$f" .build)"; cat "$f"; }; done >&2
    exit 1
fi

pids=()
for h in "${HOSTS[@]}"; do
    ssh "$h" "mkdir -p $runs && cd $runs && rm -f $out &&
        LD_LIBRARY_PATH=$CUDA_LIB exec ~/$binary --gpu $prefix" >"$logs/$h.log" 2>&1 &
    pids+=($!)
done

# Delete before pkill: the [l] stops pkill matching (and killing) this shell.
stop_all() {
    for h in "${HOSTS[@]}"; do ssh "$h" "rm -f $runs/$out; pkill -f '[l]xmf-vanity --gpu $prefix'" 2>/dev/null || true; done
    kill "${pids[@]}" 2>/dev/null || true
}
trap 'echo; echo "Stopping..."; stop_all; exit 130' INT TERM

expected=$(awk -v n="${#prefix}" 'BEGIN { printf "%.0f", 16 ^ n }')
echo "Searching for $prefix... on ${#HOSTS[@]} GPUs; about $expected tries on average."
next_report=$(( $(date +%s) + REPORT_SECONDS ))
winner=""
while [[ -z $winner ]]; do
    for i in "${!HOSTS[@]}"; do
        if ! kill -0 "${pids[$i]}" 2>/dev/null; then winner=${HOSTS[$i]}; break; fi
    done
    [[ -n $winner ]] && break
    sleep 1
    if (( $(date +%s) >= next_report )); then
        next_report=$(( next_report + REPORT_SECONDS ))
        # Sum each host's latest report ("N tries, R/s; ...").
        rate=0 tried=0
        for h in "${HOSTS[@]}"; do
            line=$(grep 'tries, ' "$logs/$h.log" | tail -1 || true)
            [[ -z $line ]] && continue
            tried=$((tried + ${line%% *}))
            r=${line#*tries, }; rate=$((rate + ${r%%/s*}))
        done
        if (( rate > 0 )); then
            awk -v t="$tried" -v r="$rate" -v e="$expected" 'BEGIN {
                left = (e - t) / r
                if (left > 0) printf "%.0f tries, %.0f/s across the GPUs; the average is about %.0f min away\n", t, r, left / 60
                else printf "%.0f tries, %.0f/s across the GPUs; past the average, still searching\n", t, r }'
        fi
    fi
done

if ! grep -q '^Found' "$logs/$winner.log"; then
    echo "$winner stopped without a result:" >&2; cat "$logs/$winner.log" >&2
    stop_all; exit 1
fi
for h in "${HOSTS[@]}"; do [[ $h != "$winner" ]] && ssh "$h" "pkill -f '[l]xmf-vanity --gpu $prefix'" 2>/dev/null || true; done
(umask 077 && scp -q "$winner:$runs/$out" "$out")
stop_all  # also deletes every remote copy
trap - INT TERM
grep -E '^(Found|About)' "$logs/$winner.log" | sed "s/^/$winner: /"
echo "Saved to $(pwd)/$out (check it with: rnid -i $out -H lxmf.delivery)"

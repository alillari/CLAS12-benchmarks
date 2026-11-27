#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_graph_noise_study.sh cohort DATA_ROOT OUTPUT
  run_graph_noise_study.sh {pilot|search} COHORT OUTPUT [DEVICE]
  run_graph_noise_study.sh {final|low_label|mid_label} COHORT OUTPUT DEVICE SEARCH_SUMMARY [M4_FEATURE_CACHE]
  run_graph_noise_study.sh plot RESULTS OUTPUT

For FM controls, set CLAS12_FM_SOURCE to the approved FM4NPP checkout.
Set GRAPH_NOISE_CONFIG to override the default study configuration.
USAGE
    exit 2
}

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo/src${CLAS12_FM_SOURCE:+:$CLAS12_FM_SOURCE}${PYTHONPATH:+:$PYTHONPATH}"
config=${GRAPH_NOISE_CONFIG:-"$repo/configs/graph_noise_cluster_v2.json"}
stage=${1:-}
case "$stage" in
    cohort)
        [[ $# == 3 ]] || usage
        exec python -m clas12_benchmarks.graph_noise_data --data-root "$2" --output "$3"
        ;;
    pilot|search)
        [[ $# == 3 || $# == 4 ]] || usage
        exec python -m clas12_benchmarks.graph_noise --phase "$stage" --config "$config" \
            --cache "$2" --output "$3" --device "${4:-cpu}"
        ;;
    final|low_label|mid_label)
        [[ $# == 5 || $# == 6 ]] || usage
        cache_options=()
        if [[ $# == 6 ]]; then
            cache_options=(--fm-features "$6")
        fi
        exec python -m clas12_benchmarks.graph_noise --phase "$stage" --config "$config" \
            --cache "$2" --output "$3" --device "$4" --selection "$5" "${cache_options[@]}"
        ;;
    plot)
        [[ $# == 3 ]] || usage
        exec python "$repo/scripts/plot_graph_noise.py" "$2" "$3"
        ;;
    *) usage ;;
esac

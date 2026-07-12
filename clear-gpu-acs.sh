#!/usr/bin/env bash
set -u

# Clear PCIe ACS Request Redirect, Completion Redirect, and Upstream Forwarding
# on every bridge in every NVIDIA GPU's upstream path. Preserve all unrelated
# ACS control bits, including Source Validation.
RETRIES=${ACS_CLEAR_RETRIES:-3}
RETRY_DELAY=${ACS_CLEAR_RETRY_DELAY:-5}
REDIRECT_MASK=$((0x001c))

log() {
  printf '%s\n' "$*"
  logger "PERF: $*" 2>/dev/null || true
}

clear_once() {
  local gpu dev parent cur after
  local seen=""
  local gpu_count=0
  local bridge_count=0
  local changed_count=0
  local failure_count=0

  for gpu in $(lspci -D -d 10de: 2>/dev/null | grep -iE 'VGA|3D controller' | awk '{print $1}'); do
    gpu_count=$((gpu_count + 1))
    dev="/sys/bus/pci/devices/$gpu"

    while [ -e "$dev" ]; do
      parent=$(basename "$(dirname "$(readlink -f "$dev")")")
      case "$parent" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f]:*) ;;
        *) break ;;
      esac
      dev="/sys/bus/pci/devices/$parent"

      case " $seen " in
        *" $parent "*) continue ;;
      esac
      seen="$seen $parent"

      cur=$(setpci -s "$parent" ECAP_ACS+0x06.w 2>/dev/null || true)
      [ -z "$cur" ] && continue
      bridge_count=$((bridge_count + 1))

      # setpci data:mask syntax clears only ACS bits 2, 3, and 4.
      if ! setpci -s "$parent" ECAP_ACS+0x06.w=0000:001c 2>/dev/null; then
        log "ACS write failed on $parent (control=$cur)"
        failure_count=$((failure_count + 1))
        continue
      fi

      after=$(setpci -s "$parent" ECAP_ACS+0x06.w 2>/dev/null || true)
      if [ -z "$after" ] || (( (0x$after & REDIRECT_MASK) != 0 )); then
        log "ACS verify failed on $parent (before=$cur after=${after:-unreadable})"
        failure_count=$((failure_count + 1))
      elif [ "$after" != "$cur" ]; then
        log "cleared ACS redirect on $parent ($cur -> $after)"
        changed_count=$((changed_count + 1))
      fi
    done
  done

  if [ "$gpu_count" -eq 0 ]; then
    log "ACS guard found no NVIDIA display GPUs"
    return 1
  fi
  if [ "$bridge_count" -eq 0 ]; then
    log "ACS guard found no ACS-capable GPU-path bridges"
    return 1
  fi
  if [ "$failure_count" -ne 0 ]; then
    log "ACS guard failed on $failure_count bridge(s)"
    return 1
  fi

  log "ACS redirect clear on $bridge_count GPU-path bridges; changed=$changed_count GPUs=$gpu_count"
  return 0
}

attempt=1
while [ "$attempt" -le "$RETRIES" ]; do
  if clear_once; then
    exit 0
  fi
  if [ "$attempt" -lt "$RETRIES" ]; then
    log "retrying ACS guard in ${RETRY_DELAY}s (attempt $attempt/$RETRIES)"
    sleep "$RETRY_DELAY"
  fi
  attempt=$((attempt + 1))
done

log "ACS redirect guard exhausted $RETRIES attempt(s)"
exit 1

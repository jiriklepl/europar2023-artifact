#!/bin/sh

[ $# = 2 ] || { echo "Usage: $0 FILE_NAME SIZE" > /dev/stderr; exit 1; }

head -c "$2" /dev/urandom > "$1"

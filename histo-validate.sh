#!/bin/sh

OUTPUT="/tmp/histo-$(uuidgen)"
OUTPUT2="/tmp/histo-$(uuidgen)"

while read -r input size; do
    find build/histo -mindepth 2 | while read -r file; do
        echo "$file" "$input" "$size"
        "$file" "$input" "$size" > "$OUTPUT"

        if [ -f "$OUTPUT2" ]; then
            diff "$OUTPUT" "$OUTPUT2" || exit 1
        fi

        mv "$OUTPUT" "$OUTPUT2"
    done || exit 1

    rm "$OUTPUT2"
done <<EOF
build/histo/text 131072
EOF

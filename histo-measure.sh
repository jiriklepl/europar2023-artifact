#!/bin/sh

OUTPUT=out/histo-$(uname -n)-$(uuidgen).csv
REPS=10

mkdir -p out

echo "running measurements (outputting to '$OUTPUT'):" 1>&2

while read -r input size; do
	echo "$(date +%H:%M:%S:) running measurements on '$input' (size: $size):" 1>&2

	if [ -f "$input" ]; then
		for _ in $(seq "$REPS"); do
			find build/histo -type f -mindepth 2 | shuf | while read -r file; do
				printf "%s" "$file,$input,$size,$(uname -n),$(date)," >> "$OUTPUT"

				"$file" "$input" "$size" > /dev/null 2>> "$OUTPUT"
			done
		done
	else
		echo "warning: requesting measurements on nonexistent input" 1>&2
	fi
done <<EOF
build/histo/text 131072
build/histo/text 524288
build/histo/text 2097152
build/histo/text 8388608
build/histo/text 33554432
build/histo/text 134217728
build/histo/text 536870912
build/histo/text 2147483648
EOF

#!/bin/sh

OUTPUT=out/kmeans-$(uname -n)-$(uuidgen).csv
REPS=10

mkdir -p out

echo "running measurements (outputting to '$OUTPUT'):" 1>&2

while read -r input size clusters dims; do
	echo "$(date +%H:%M:%S:) running measurements on '$input' (size: $size):" 1>&2

	if [ -f "$input" ]; then
		for _ in $(seq "$REPS"); do
			find build/kmeans -type f -mindepth 2 | shuf | while read -r file; do
				printf "%s" "$file,$input,$size,$(uname -n),$(date)," >> "$OUTPUT"

				"$file" "$size" "$clusters" "$dims" < "$input" > /dev/null 2>> "$OUTPUT"
			done
		done
	else
		echo "warning: requesting measurements on nonexistent input" 1>&2
	fi
done <<EOF
build/kmeans/kmeans_7_4_20000 20000 7 4
build/kmeans/kmeans_7_6_20000 20000 7 6
build/kmeans/kmeans_10_3_20000 20000 10 3
EOF

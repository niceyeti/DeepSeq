#!/bin/sh

# Description: the fbdata files consist of lines, each of which is a python
# tuple whose [0] member is the link and [1] is a json object. To extract and
# output any field, the task is simply to read the tuple with python, pipe it to
# jq with any transformer expression, and then pipe that output to whatever file
# you want. Currently I'm not using jq, just pulling out the 'description' for
# natural language data.
# 
# Use this command line shell apprach instead of python as it is more
# productive.

if [ -z "$FB_DATA_PATH" ]; then
    echo "FB_DATA_PATH required but not passed."
    exit 1
fi
if [ ! -f "$FB_DATA_PATH" ]; then
    echo "FB_DATA_PATH not found or not a file: $FB_DATA_PATH"
    exit 1
fi


while IFS= read -r line; do
    python -c "print($line[1]['og_object']['description']) if 'og_object' in $line[1] and 'description' in $line[1]['og_object'] else ''"
done < "$FB_DATA_PATH"

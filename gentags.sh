#!/bin/bash
rm TAGS
find . -name "*.[ch]" -exec etags -a '{}' \;
find . -name "*.cu" -exec etags -a '{}' \;
find . -name "*.cuh" -exec etags -a '{}' \;
find . -name "*.cpp" -exec etags -a '{}' \;

#!/bin/sh
url=$1
wc $url

curl -X GET $url \
-H "Accept: application/json" \
-H "Accept: application/json" -H "Authorization: Bearer ${Otoken}"

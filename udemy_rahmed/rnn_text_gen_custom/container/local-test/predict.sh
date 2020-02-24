#!/bin/bash

payload=$1
content=${2:-application/x-image}

curl -d @${payload} -H "Content-Type: ${content}" http://localhost:8080/invocations
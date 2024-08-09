#!/bin/bash

while true
do
  echo "docker system df"
  docker system df
  echo "df -kh ."
  df -kh .
  echo "du -h --max-depth=1 ."
  du -h --max-depth=1 .
  sleep 1m
done
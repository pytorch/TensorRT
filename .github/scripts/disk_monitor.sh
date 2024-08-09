#!/bin/bash

while true
do
  echo "docker system df"
  docker system df
  echo "df -kh ."
  df -kh .
  sleep 1m
done
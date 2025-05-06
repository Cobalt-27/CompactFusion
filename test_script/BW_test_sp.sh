#!/bin/bash

# --- 定义要测试的带宽速率列表 ---
# TARGET_RATES=("100mbit" "500mbit" "1gbit" "2gbit" "5gbit" "10gbit" "20gbit")
# "300MB, 400MB,500MB * 4 * 8
# "9600mbit" "12800mbit" 
TARGET_RATES=("16000mbit" "32000mbit" "48000mbit" "unlimited")
# TARGET_RATES=("16gbit")
# TARGET_RATES=("8gbit")
# TEST_MODES=("ring" "df" "binary" "lowrank16")

TEST_MODES=("pipe" "binary" "lowrankq32")
# TEST_MODES=("pipe" "ulysses" "patch" "int2patch"  "ring" "df")

TEST_LOOP=11

RUN_SCRIPT_PATH="/workspace/xDiT/examples/run_BWTest.sh" 

# --- 循环测试不同的带宽限制 ---
for TEST_MODE in "${TEST_MODES[@]}"
do
  for TARGET_RATE in "${TARGET_RATES[@]}"
  do
    echo "=========================================="
    echo "--- Starting test with TEST_MODE=${TEST_MODE} TARGET_RATE=${TARGET_RATE} ---"
    echo "=========================================="
    
    bash ${RUN_SCRIPT_PATH} ${TEST_MODE} ${TARGET_RATE} ${TEST_LOOP}
    
    if [ $? -ne 0 ]; then
      echo "!!! Error occurred during test with TEST_MODE=${TEST_MODE} TARGET_RATE=${TARGET_RATE} !!!"
    fi
    
    echo "--- Test finished for TEST_MODE=${TEST_MODE} TARGET_RATE=${TARGET_RATE} ---"
    echo "Pausing for 5 seconds..."
    sleep 5
  done
done

echo "========================================================"
echo "--- All bandwidth tests completed successfully. ---"
echo "========================================================"

exit 0
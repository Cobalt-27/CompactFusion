import os

TEST_ENABLE = bool(os.environ.get('COMPACT_TEST_ENABLE', False))
TEST_MODEL = os.environ.get('COMPACT_TEST_MODEL', None)
TEST_METHOD = os.environ.get('COMPACT_TEST_METHOD', None)
TEST_LOOP = int(os.environ.get('COMPACT_TEST_LOOP', 20))

#export TQDM_DISABLE=1 to disable progress bar
#export LOG_LEVEL=warning to disable verbose info

def test_hello():
    print('-'*20)
    print(f"🟩  Test enabled" if TEST_ENABLE else "🟧 Test disabled")
    if TEST_ENABLE:
        print(f"➡️  TEST_MODEL: {TEST_MODEL}")
        print(f"➡️  TEST_METHOD: {TEST_METHOD}")
        print(f"➡️  TEST_LOOP: {TEST_LOOP}")
        print('-'*20)


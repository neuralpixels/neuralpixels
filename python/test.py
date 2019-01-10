import os
from neuralpixel_tests import Tester

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    print('Starting tests')
    tester = Tester()
    tester.run_all()
    print('Test complete')
    print('PASSED: {}'.format(tester.num_passed))
    print('FAILED: {}'.format(tester.num_failed))
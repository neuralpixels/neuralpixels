import os
from neuralpixels.testing import image
from neuralpixels.testing import layers
test_root = os.path.dirname(__file__)


class Tester(object):
    def __init__(self):
        self.num_passed = 0
        self.num_failed = 0

    def _run_test(self, test_function):
        num_passed, num_failed = test_function()
        self.num_passed += num_passed
        self.num_failed += num_failed
        print('')

    def run_all(self):
        self._run_test(image.run_filters_test)
        self._run_test(layers.run_prelu_clipped_test)


def run_all_tests():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print('Starting tests')
    tester = Tester()
    tester.run_all()
    print('Test complete')
    print('PASSED: {}'.format(tester.num_passed))
    print('FAILED: {}'.format(tester.num_failed))

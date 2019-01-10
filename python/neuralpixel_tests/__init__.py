import os
from neuralpixel_tests import image
test_root = os.path.dirname(__file__)


class Tester(object):
    def __init__(self):
        self.num_passed = 0
        self.num_failed = 0

    def _run_test(self, test_function):
        num_passed, num_failed = test_function()
        self.num_passed += num_passed
        self.num_failed += num_failed

    def run_all(self):
        self._run_test(image.run_filters_test)



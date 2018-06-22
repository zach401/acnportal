import TestCase
from datetime import datetime

class Simulator:

    def __init__(self, test_case):
        self.iteration = 0
        self.test_case = test_case
        pass

    def run(self):
        pass


if __name__ == '__main__':
    print("hej")
    test_case = TestCase.generate_test_case_local('test_session.p',
                                                  datetime.strptime("21/11/06", "%d/%m/%y"),
                                                  datetime.strptime("21/11/22", "%d/%m/%y"))
    sim = Simulator(test_case)
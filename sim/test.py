from acnlib import TestCase
from acnlib.Garage import Garage
from datetime import datetime
from acnlib import StatModel

#tc = TestCase.generate_test_case_local()
if __name__ == '__main__':

    garage = Garage()
    test_case_model = garage.generate_test_case(datetime.strptime("01/06/18", "%d/%m/%y"),
                                                datetime.strptime("30/06/18", "%d/%m/%y"),
                                                period=1)
    test_case_real = TestCase.generate_test_case_local('July_25_Sessions.pkl',
                                                       datetime.strptime("01/06/18", "%d/%m/%y"),
                                                       datetime.strptime("30/06/18", "%d/%m/%y"),
                                                       period=1)
    StatModel.compare_model_to_real(test_case_real, test_case_model)
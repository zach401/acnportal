from unittest import TestCase
from acnportal.signals.tariffs.tou_tariff import TimeOfUseTariff
from datetime import datetime


class TestTimeOfUseTariff(TestCase):
    def test_get_tariff_middle_mid_peak(self):
        tariff = TimeOfUseTariff("sce_tou_ev_4_march_2019")
        self.assertEqual(tariff.get_tariff(datetime(2019, 7, 1, 9, 30)), 0.0925)

    def test_get_tariff_edge_mid_peak(self):
        tariff = TimeOfUseTariff("sce_tou_ev_4_march_2019")
        self.assertEqual(tariff.get_tariff(datetime(2019, 7, 1, 8)), 0.0925)

    def test_get_tariffs_all_peak(self):
        tariff = TimeOfUseTariff("sce_tou_ev_4_march_2019")
        tariff_list = tariff.get_tariffs(datetime(2019, 7, 1, 12), 12, 5)
        self.assertListEqual(tariff_list, [0.26668] * 12)

    def test_get_tariffs_period_boundary(self):
        tariff = TimeOfUseTariff("sce_tou_ev_4_march_2019")
        tariff_list = tariff.get_tariffs(datetime(2019, 7, 1, 7), 24, 5)
        self.assertListEqual(tariff_list, [0.05623] * 12 + [0.0925] * 12)

    def test_get_tariffs_weekday_weekend(self):
        tariff = TimeOfUseTariff("sce_tou_ev_4_march_2019")
        length = 9 * 12
        tariff_list = tariff.get_tariffs(datetime(2019, 7, 5, 23, 30), length, 5)
        self.assertListEqual(tariff_list, [0.05623] * length)

    def test_non_integer_period_boundaries(self):
        tariff = TimeOfUseTariff("pge_a10_tou_aug_2019")
        tariff_list = tariff.get_tariffs(datetime(2019, 7, 1, 8, 25), 3, 5)
        self.assertListEqual(tariff_list, [0.14903, 0.1771, 0.1771])

    def test_get_tariffs_season_boundary(self):
        tariff = TimeOfUseTariff("sce_tou_ev_4_march_2019")
        tariff_list = tariff.get_tariffs(datetime(2019, 9, 30, 23), 24, 5)
        self.assertListEqual(tariff_list, [0.05623] * 12 + [0.06087] * 12)

    def test_tariff_not_start_at_0(self):
        with self.assertRaises(ValueError):
            _ = TimeOfUseTariff("invalid_tariff_schedule", tariff_dir="tests")

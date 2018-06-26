import matplotlib.pyplot as plt

class GraphDrawer:

    def __init__(self, simulator):
        self.simulator = simulator

    def draw_charge_rates(self):
        test_case_data = self.simulator.get_simulation_data()
        for ev_id, data in test_case_data.items():
            t = []
            value = []
            for sample in data:
                t.append(sample['time'])
                value.append(sample['charge_rate'])
            plt.plot(t, value)
        plt.show()

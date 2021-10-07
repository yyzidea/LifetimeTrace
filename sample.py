from python.driver import TimeTagger
from lifetime_trace import LifetimeTrace
from time_tagger_utility import *

tagger = TimeTagger.createTimeTagger()
meas = LifetimeTrace(2, 5, 100, 10000, int_time=0.1, tagger=tagger)
hist = TimeTagger.Histogram(tagger, 2, 5, 100, 10000)

cv = wave_generator.CyclicVoltammetry('COM4', -10, 10, scan_rate=1)

# Cyclic voltammetry lifetime trace measurement
cv.start()
meas.startFor(400)
plot_lifetime_trace(meas)
cv.stop()
meas.stop()

# Save data
data_dir = r'E:\yyz\Quantum Dot Device\Lifetime trace\20210915'
dot_num = 6
power = '70nw'
save_data('%s\\lll_%d_device_lifetimetrace_cw_450nm_%s_-10V_10V.csv' % (data_dir, dot_num, power), meas)
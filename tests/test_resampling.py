import unittest
import numpy as np
from voicesnake.resampling import StreamingResampler


class ResamplingTests(unittest.TestCase):
    def test_native_passthrough(self):
        x = np.linspace(-1, 1, 1000, dtype=np.float32)
        np.testing.assert_array_equal(StreamingResampler(16000).process(x), x)

    def test_streaming_matches_whole_without_drift(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                x = np.sin(2*np.pi*440*np.arange(rate*2)/rate).astype(np.float32)
                expected = StreamingResampler(rate).process(x)
                stream = StreamingResampler(rate)
                actual = np.concatenate([stream.process(x[i:i+317]) for i in range(0, len(x), 317)])
                self.assertEqual(len(actual), 32000)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                self.assertGreater(np.sqrt(np.mean(actual[100:]**2)), .69)
                self.assertLess(np.sqrt(np.mean(actual[100:]**2)), .72)

    def test_alias_rejection(self):
        for rate in (44100, 48000):
            x = np.sin(2*np.pi*12000*np.arange(rate)/rate).astype(np.float32)
            y = StreamingResampler(rate).process(x)
            self.assertLess(np.sqrt(np.mean(y[100:]**2)), .01)


if __name__ == '__main__':
    unittest.main()

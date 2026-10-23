import unittest
from voicesnake.game import Snake
from voicesnake.detector import Detector


class CoreTests(unittest.TestCase):
    def test_reverse_and_multiple_commands(self):
        g = Snake()
        self.assertFalse(g.turn('left'))
        self.assertTrue(g.turn('up'))
        self.assertFalse(g.turn('left'))
        self.assertEqual(g.direction, 'right')
        g.tick()
        self.assertEqual(g.direction, 'up')

    def test_food_and_wrap(self):
        g = Snake()
        x,y = g.body[0]
        g.food = (x+1,y)
        g.tick()
        self.assertEqual((g.score,len(g.body)), (1,4))
        g.body=[(23,5),(22,5),(21,5)]
        g.food=(12,12)
        g.tick()
        self.assertEqual(g.body[0],(0,5))
        self.assertFalse(g.over)

    def test_tail_is_vacated(self):
        g=Snake()
        g.body=[(2,2),(2,3),(1,3),(1,2)]
        g.direction=g.pending_direction='left'
        g.food=(8,8)
        g.tick()
        self.assertFalse(g.over)

    def test_pause(self):
        g=Snake(); g.paused=True
        before=list(g.body)
        self.assertFalse(g.turn('up'))
        g.tick()
        self.assertEqual(g.body,before)

    def test_gate_latch_and_release(self):
        d=Detector(threshold=.8,confirmations=2)
        self.assertIsNone(d.update('up',.99,False,0))
        self.assertIsNone(d.update('yes',.99,True,1))
        self.assertIsNone(d.update('up',.6,True,1.1))
        self.assertIsNone(d.update('up',.99,True,1.2))
        self.assertEqual(d.update('up',.99,True,1.3).direction,'up')
        self.assertIsNone(d.update('up',.99,True,2))
        self.assertIsNone(d.update('right',.99,True,2.1))
        self.assertEqual(d.update('right',.99,True,2.2).direction,'right')
        d.update('up',.99,False,3)
        d.update('up',.99,False,3.4)
        self.assertIsNone(d.update('right',.99,True,4))
        self.assertEqual(d.update('right',.99,True,4.1).direction,'right')



class AudioTests(unittest.TestCase):
    def test_buffer_and_toggle_generation(self):
        import numpy as np
        from voicesnake.audio import AudioWorker
        worker=AudioWorker()
        worker.callback(np.ones((320,1),dtype=np.float32),320,None,None)
        self.assertEqual(worker.filled,320)
        self.assertTrue(np.all(worker.buffer[-320:]==1))
        self.assertTrue(np.all(worker.buffer[:-320]==0))
        worker.set_enabled(False)
        generation=worker.generation
        worker.set_enabled(False)
        self.assertEqual(worker.generation,generation)
        worker.set_enabled(True)
        self.assertEqual(worker.generation,generation+1)

    def test_replay_silence_and_noise_gate(self):
        import numpy as np
        from voicesnake.evaluate import replay
        class NeverInfer:
            def predict(self, samples):
                raise AssertionError('Below-threshold audio must skip inference')
        samples=np.random.default_rng(1).normal(0,.001,160000).astype(np.float32)
        commands,times=replay(NeverInfer(),samples,.8,.008,3)
        self.assertEqual(commands,[])
        self.assertEqual(times,[])

if __name__ == '__main__':
    unittest.main()

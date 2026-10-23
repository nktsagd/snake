import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PortabilityTests(unittest.TestCase):
    def test_model_is_packaged_and_checksum_verified(self):
        from voicesnake.model import CHECKPOINT,verify_assets,Recognizer
        import numpy as np
        self.assertEqual(CHECKPOINT.parent.name,'assets')
        self.assertEqual(verify_assets()['classes'],['up','right','left','down'])
        label,confidence,ms=Recognizer().predict(np.zeros(16000,dtype=np.float32))
        self.assertIn(label,['up','right','left','down'])
        self.assertTrue(0 <= confidence <= 1)
        self.assertGreaterEqual(ms,0)

    def test_output_path_can_be_overridden_independent_of_cwd(self):
        from voicesnake.paths import data_home
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ,{'VOICESNAKE_HOME':directory}):
                self.assertEqual(data_home(),Path(directory).resolve())

if __name__=='__main__': unittest.main()

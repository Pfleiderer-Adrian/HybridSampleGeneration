import json,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from hybrid_sample_generator.configuration.root import Configuration
from hybrid_sample_generator.configuration.postprocessing import PostprocessingConfiguration
from hybrid_sample_generator.postprocessing.foreground import object_mask
from hybrid_sample_generator.postprocessing.service import PostprocessingService,decision_for
from hybrid_sample_generator.persistence.artifact_store import ArtifactStore
from hybrid_sample_generator.persistence.study_repository import StudyRepository
from hybrid_sample_generator.domain.records import OriginalSample,HybridSample

class PostprocessingTests(unittest.TestCase):
    def test_configuration_roundtrip_and_legacy(self):
        cfg=Configuration('test');cfg.postprocessing.enabled=True
        values=cfg.to_dict();self.assertTrue(Configuration.from_dict(values).postprocessing.enabled)
        del values['postprocessing'];self.assertFalse(Configuration.from_dict(values).postprocessing.enabled)
        cfg.postprocessing.rules.mask_area_fraction.minimum=.8
        cfg.postprocessing.rules.mask_area_fraction.maximum=.2
        with self.assertRaises(ValueError):cfg.validate()

    def test_foreground_can_wallplugs(self):
        c=PostprocessingConfiguration().foreground
        a=np.full((3,50,70),255,dtype='uint8');a[:,10:40,10:60]=100;a[:,20:25,20:25]=255;a[:,0:2,0:2]=0
        m=object_mask(a,c);self.assertTrue(m[22,22]);self.assertFalse(m[0,0]);self.assertEqual(m.sum(),1500)
        c.channel_reduction='grayscale';c.comparison='above';c.threshold=40;c.components='all';c.min_component_area=10;c.fill_holes=False
        a=np.zeros((1,50,70));a[:,5:15,5:15]=100;a[:,30:40,40:50]=100;a[:,0,0]=100
        m=object_mask(a,c);self.assertEqual(m.sum(),200)

    def test_decisions_exclusion_reporting_rerun_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=ArtifactStore(tmp);repo=StudyRepository(Path(tmp)/'artifacts.sqlite')
            a=np.full((3,50,70),255,dtype='uint8');a[:,10:40,10:60]=100
            path=store.save_entity_array('original_samples','o','image',a)
            original=OriginalSample('o','test',path,None,2,False,False,0);repo.replace_original_samples([original])
            for name,point in [('good',(20,20)),('bad',(0,0))]:
                mask=np.zeros((1,50,70));mask[0,point[0],point[1]]=1
                mp=store.save_entity_array('hybrid_samples',name,'segmentation',mask)
                ip=store.save_entity_array('hybrid_samples',name,'image',a)
                repo.upsert_hybrid_sample(HybridSample(name,'o',0 if name=='good' else 1,ip,mp,'generated'))
            cfg=PostprocessingConfiguration(enabled=True,action='exclude',save_previews=True)
            cfg.rules.minimum_object_overlap.enabled=True
            result=PostprocessingService(repo,store,cfg).run();self.assertEqual(result['excluded'],1)
            bad=repo.get_hybrid_sample('bad');good=repo.get_hybrid_sample('good')
            self.assertFalse(decision_for(repo,store,bad,original,cfg)['accepted'])
            from examples.mvtec_ad2.downstream.datasets import HybridPairs
            from examples.mvtec_ad2.downstream.configuration import DataConfiguration
            with patch('examples.mvtec_ad2.downstream.datasets.verify_repository'), \
                 patch.object(repo,'list_placements',return_value=[SimpleNamespace(synthetic_anomaly_id='s',id='p')]), \
                 patch.object(repo,'get_synthetic_anomaly',return_value=SimpleNamespace(real_anomaly_id='r')), \
                 patch.object(repo,'get_real_anomaly',return_value=SimpleNamespace(original_sample_id='o')):
                pairs=HybridPairs(repo,store,{},DataConfiguration(),cfg)
                self.assertEqual(len(pairs),1)
                self.assertEqual(pairs.provenance[0]['hybrid_id'],'good')
            self.assertEqual(len(repo.list_hybrid_samples()),2)
            self.assertTrue(store.resolve(bad.image_path).exists())
            self.assertEqual(len(list((Path(result['report_folder'])/'previews').glob('*.png'))),2)
            cfg.action='report'
            with self.assertRaisesRegex(ValueError,'stale'):decision_for(repo,store,good,original,cfg)
            result=PostprocessingService(repo,store,cfg).run();self.assertEqual(result['excluded'],0);self.assertEqual(result['failed'],1)
            cfg.rules.minimum_object_overlap.enabled=False
            result=PostprocessingService(repo,store,cfg).run();self.assertEqual(result['failed'],0)
            store.save_entity_array('hybrid_samples','good','segmentation',np.ones((1,50,70)))
            with self.assertRaisesRegex(ValueError,'stale'):decision_for(repo,store,good,original,cfg)
            repo.replace_hybrid_plan([],[])
            with repo.connection() as db:self.assertEqual(db.execute('select count(*) from postprocessing_decisions').fetchone()[0],0)

    def test_rules_and_empty_foreground_fail_safely(self):
        from hybrid_sample_generator.postprocessing.measurements import measure
        from hybrid_sample_generator.postprocessing.rules import violations
        c=PostprocessingConfiguration();c.rules.mask_area_fraction.enabled=True;c.rules.mask_area_fraction.maximum=.1
        self.assertIn('mask_area_fraction',violations(measure(np.ones((1,5,5))),c.rules))
        self.assertIn('empty_mask',violations(measure(np.zeros((1,5,5))),c.rules))
        with self.assertRaisesRegex(ValueError,'empty object'):measure(np.ones((1,5,5)),np.zeros((5,5),bool))

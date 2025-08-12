from services import Logger
from addons import ArchSmith, ASReport, ASTracer
from schemas import Batch, Artefact

class HFCatalogueIntegrator:
    @staticmethod
    def integrate(batch: Batch=None, artefacts: list[Artefact]=None):
        if not isinstance(batch, Batch) and not isinstance(artefacts, list):
            Logger.log("HFCATALOGUEINTEGRATOR INTEGRATE ERROR: Terminating due to invalid input.")
            return
        elif artefacts and not all(isinstance(a, Artefact) for a in artefacts):
            Logger.log("HFCATALOGUEINTEGRATOR INTEGRATE ERROR: Terminating due to invalid artefacts.")
            return
        
        tracer = ArchSmith.newTracer("HF Catalogue Integration for {}".format("Batch {}".format(batch.id) if batch else "{} Artefacts".format(len(artefacts))))
        tracer.start()
        
        tracer.addReport(ASReport("HFCATALOGUEINTEGRATOR", "Dummy Report"))
        
        tracer.end()
        ArchSmith.persist()
from typing import List
from services import Logger
from addons import ArchSmith, ASReport
from schemas import Batch, Category, BatchArtefact
from autoCategorise import AutoCategoriser

class HFCatalogueIntegrator:
    @staticmethod
    def integrate(batch: Batch):
        """Two steps:
        1. Auto-categorisation
        2. Auto Profiling
        
        Provide a batch. At the end, all the artefacts will be `BatchArtefact.Status.INTEGRATED` and the batch will be `Batch.Stage.COMPLETED`.
        
        Uses ArchSmith tracing for transparency.

        Args:
            batch (Batch, optional): Batch to integrate. Will only process artefacts with status `BatchArtefact.Status.CONFIRMED`. Defaults to None.
        """
        
        if not isinstance(batch, Batch):
            Logger.log("HFCATALOGUEINTEGRATOR INTEGRATE ERROR: Terminating due to invalid input.")
            return
        
        job = batch.job
        job.start()
        job.save()

        tracer = ArchSmith.newTracer("HF Catalogue Integration for Batch {}".format(batch.id))
        tracer.start()
        
        try:
            batch.loadAllArtefacts()
        except Exception as e:
            tracer.addReport(
                ASReport(
                    "HFCATALOGUEINTEGRATOR INTEGRATE ERROR",
                    "Failed to load artefacts for batch {}: {}".format(batch.id, e)
                )
            )
            return
        
        targetHFBatchArtefacts: List[BatchArtefact] = []
        try:
            for batchArt in batch.artefacts.values():
                if batchArt.artefact.artType == "mm":
                    batchArt.artefact = None
                    batchArt.integrationResult = "No integration required for meeting minute artefact."
                    batchArt.stage = BatchArtefact.Status.INTEGRATED
                    batchArt.save()
                else:
                    batchArt.artefact = None
                    targetHFBatchArtefacts.append(batchArt)
        except Exception as e:
            tracer.addReport(
                ASReport(
                    "HFCATALOGUEINTEGRATOR INTEGRATE ERROR",
                    "Failed to process loaded batch artefacts for MM integration and HF preparation; error: {}".format(e)
                )
            )
            return
        
        # Auto-categorisation
        categories: List[Category] = Category.load()
        for batchArt in targetHFBatchArtefacts:
            try:
                try:
                    batchArt.getArtefact(includeMetadata=True)
                except Exception as e:
                    tracer.addReport(
                        ASReport(
                            "HFCATALOGUEINTEGRATOR INTEGRATE WARNING",
                            "Skipping auto-categorisation for artefact {} due to error in load: {}".format(batchArt.id, e)
                        )
                    )
                    continue
                
                targetArtefact = batchArt.artefact
                result = AutoCategoriser.feed(targetArtefact, categories, tracer)
                
                tracer.addReport(
                    ASReport(
                        "HFCATALOGUEINTEGRATOR INTEGRATE{}".format(" WARNING" if result is None else ""),
                        "Auto-categorisation result for artefact {}: {}".format(targetArtefact.id, result)
                    )
                )
                
                batchArt.artefact = None
                
                if not isinstance(batchArt.integrationResult, dict):
                        batchArt.integrationResult = {}
                if result is None:
                    batchArt.integrationResult['autocategoriser'] = "Could not automatically categorise artefact."
                else:
                    targetCat = [x for x in categories if x.id == result]
                    if len(targetCat) == 0:
                        targetCat = 'Unknown'
                    else:
                        targetCat = targetCat[0].name
                    batchArt.integrationResult['autocategoriser'] = "Assigned to category with name '{}'.".format(targetCat)
                
                batchArt.stage = BatchArtefact.Status.INTEGRATED
                batchArt.save()
            except Exception as e:
                tracer.addReport(
                    ASReport(
                        "HFCATALOGUEINTEGRATOR INTEGRATE ERROR",
                        "Failed to auto-categorise artefact {}: {}".format(batchArt.id, e)
                    )
                )
                continue
        
        try:
            batch.stage = Batch.Stage.COMPLETED
            batch.save()
        except Exception as e:
            tracer.addReport(
                ASReport(
                    "HFCATALOGUEINTEGRATOR INTEGRATE ERROR",
                    "Failed to mark batch {} as completed: {}".format(batch.id, e)
                )
            )
        
        job.end()
        job.save()
        
        tracer.addReport(
            ASReport(
                "HFCATALOGUEINTEGRATOR INTEGRATE",
                "Completed integration procedure for batch {}.".format(batch.id)
            )
        )
        
        tracer.end()
        ArchSmith.persist()
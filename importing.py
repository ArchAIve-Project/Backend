from models import Batch, Artefact
from services import ThreadManager

class ImportProcessor:
    @staticmethod
    def processBatch(batch: Batch, chunk_size=10):
        batch.getArtefacts(unprocessed=True)
        unprocessedItems = [a for a in batch.unprocessed.values() if isinstance(a, Artefact)]
        total = len(unprocessedItems)
        i = 0
        failed_ids = set()

        while i < total:
            currentChunk = unprocessedItems[i:i + chunk_size]
            ThreadManager.defaultProcessor.addJob(ImportProcessor.processChunk, currentChunk, batch, failed_ids)

            # Wait until all items in currentChunk are either processed or failed
            while not all(
                artefact.id in batch.processed or artefact.id in failed_ids  # try batch.where() ?
                for artefact in currentChunk if isinstance(artefact, Artefact)
            ) and not batch.cancelled:
                time.sleep(0.5)

            if batch.cancelled:
                break
            i += chunk_size
        
        batch.processingJob = None
        return True

    @staticmethod
    def processChunk(chunk, batch, failed_ids: set):
        for artefact in chunk:
            try:
                ImportProcessor.processItem(artefact)
                batch.reload()
                batch.move(artefact, Batch.Stage.PROCESSED)
                batch.save()
            except Exception as e:
                Logger.log(f"Failed to process artefact {artefact.id}: {e}")
                failed_ids.add(artefact.id)

            if batch.cancelled:
                return

    @staticmethod
    def processItem(artefact: Artefact):
        out = MetadataGenerator.generate(artefact.image)
        if isinstance(out, str):
            raise Exception("IMPORTPROCESSOR PROCESSITEM ERROR: Unexpected output from MetadataGenerator when processing artefact '{}'; response: {}".format(artefact.id, out))
        artefact.metadata = Metadata.fromMetagen(artefact.id, out)
        artefact.save()
        return artefact
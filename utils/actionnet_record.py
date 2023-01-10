from .video_record import VideoRecord


class ActionNetVideoRecord(VideoRecord):
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf

    @property
    def start_frame(self):
        return self._series['start_frame']

    @property
    def end_frame(self):
        return self._series['stop_frame']

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame}

    @property
    def label(self):
        if 'verb_class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['verb_class']

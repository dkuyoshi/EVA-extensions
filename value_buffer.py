class ValueBuffer(object):
    """Buffer for outputting nonparametric q value"""
    def __init__(self, capacity, k,  key_width, num_actions, xp, LRU=False):
        self.capacity = capacity
        self.neighbors = k
        self.key_width = key_width
        self.num_actions = num_actions
        self.xp = xp
        self.LRU = LRU
        self.xp = xp

        if self.LRU:
            self.lru_timestamps = self.xp.empty(self.capacity, dtype=self.xp.int32)
            self.current_timestamp = 0
            self.i = 0
            self.embeddings = self.xp.empty((self.capacity, key_width), dtype=self.xp.float32)
            self.values = self.xp.empty((self.capacity, num_actions), dtype=self.xp.float32)
        else:
            self.embeddings = self.xp.empty((0, key_width), dtype=self.xp.float32)
            self.values = self.xp.empty((0, num_actions), dtype=self.xp.float32)

    def store(self, embeddings, values):
        if self.LRU:
            self._lru_store(embeddings, values)
        else:
            self._normal_store(embeddings, values)

    def _normal_store(self, embeddings, values):
        """Save Q value corresponding to emmbedding
        """
        self.embeddings = self.xp.concatenate((self.embeddings, embeddings), axis=0)[-self.capacity:]
        self.values = self.xp.concatenate((self.values, values), axis=0)[-self.capacity:]

    def _lru_store(self, embeddings, values):
        for embedding, value in zip(embeddings, values):
            if self.i < self.capacity:
                self.embeddings[self.i] = embedding
                self.values[self.i] = value
                self.lru_timestamps[self.i] = self.current_timestamp
                self.i += 1
                self.current_timestamp += 1
            else:
                index = self.xp.argmin(self.lru_timestamps)
                self.embeddings[index] = embedding
                self.values[index] = value
                self.lru_timestamps[index] = self.current_timestamp
                self.current_timestamp += 1
        return

    def get_q(self, embedding):
        indices = self.index_search(embedding)
        if self.LRU:
            for index in indices:
                self.lru_timestamps[index] = self.current_timestamp
                self.current_timestamp += 1
        values = self.xp.asarray(self.values[indices])
        return self.xp.mean(values, axis=0)

    def index_search(self, embedding):
        distances = self.xp.sum((self.xp.asarray(embedding) - self.embeddings)**2, axis=1)
        return self.xp.argsort(distances)[:self.neighbors]
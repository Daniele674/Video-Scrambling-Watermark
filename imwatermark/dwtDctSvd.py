import numpy as np
import cv2
import pywt
import pprint
from scipy.fftpack import dct, idct

pp = pprint.PrettyPrinter(indent=2)


class EmbedDwtDctSvd(object):
    def __init__(self, watermarks=[], wmLen=8, scales=[0, 36, 0], block=4):     #108
        self._watermarks = np.array(watermarks)  # converto in array per operazioni vettoriali
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

    def encode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # Per i canali 0 e 1 (tipicamente Y e U oppure Y e V)
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            # Assicuriamoci che la dimensione sia multiplo di self._block
            proc_row = row - (row % self._block)
            proc_col = col - (col % self._block)
            coeffs = pywt.dwt2(yuv[:proc_row, :proc_col, channel], 'haar')
            ca1, (h1, v1, d1) = coeffs

            # Elaborazione vettorializzata dei blocchi
            ca1 = self.encode_frame(ca1, self._scales[channel])

            # Ricostruzione con la IDWT
            yuv[:proc_row, :proc_col, channel] = pywt.idwt2((ca1, (h1, v1, d1)), 'haar')

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # Lista di liste per raccogliere i punteggi per ciascun bit
        scores = [[] for _ in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            proc_row = row - (row % self._block)
            proc_col = col - (col % self._block)
            coeffs = pywt.dwt2(yuv[:proc_row, :proc_col, channel], 'haar')
            ca1, _ = coeffs

            scores = self.decode_frame(ca1, self._scales[channel], scores)

        # Media dei punteggi per ciascun bit
        avgScores = list(map(lambda l: np.array(l).mean() if len(l) > 0 else 0, scores))
        bits = (np.array(avgScores) * 255 > 127)
        return bits

    def encode_frame(self, frame, scale):
        """
        Elabora il frame (matrice M x N) suddividendolo in blocchi di dimensione self._block x self._block,
        e inserisce in maniera vettorializzata il watermark.
        """
        block = self._block
        rows, cols = frame.shape
        n_blocks_row = rows // block
        n_blocks_col = cols // block
        total_blocks = n_blocks_row * n_blocks_col

        # Reshape in blocchi: forma (n_blocks_row, n_blocks_col, block, block)
        blocks = frame[:n_blocks_row * block, :n_blocks_col * block].reshape(n_blocks_row, block, n_blocks_col, block)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block, block)

        # Calcola la 2D DCT in modalità vettoriale:
        # Applichiamo la DCT lungo l'asse 1 e poi lungo l'asse 2, con normalizzazione "ortho"
        blocks_dct = dct(dct(blocks, axis=1, norm='ortho'), axis=2, norm='ortho')

        # Calcola la SVD in batch per ciascun blocco
        # np.linalg.svd supporta array impilati (batch) a partire da NumPy 1.8
        U, s, Vh = np.linalg.svd(blocks_dct, full_matrices=False)

        # Determina, per ogni blocco, il bit da inserire: viene preso dal watermark ciclicamente
        # Creiamo un array dei bit con dimensione (total_blocks,)
        wm_bits = np.tile(self._watermarks, int(np.ceil(total_blocks / self._wmLen)))[:total_blocks]
        # Assicuriamoci che i bit siano numerici (0 o 1)
        wm_bits = wm_bits.astype(np.float64)

        # Modifica il primo valore singolare per ciascun blocco in base al watermark
        # La quantizzazione: s0' = ( floor(s0/scale) + 0.25 + 0.5 * wm_bit ) * scale
        s0 = s[:, 0]
        s0_mod = (np.floor(s0 / scale) + 0.25 + 0.5 * wm_bits) * scale
        s[:, 0] = s0_mod

        # Ricostruisce i blocchi DCT modificati: u * diag(s) * Vh
        # Utilizziamo la moltiplicazione elemento per elemento: moltiplichiamo ogni colonna di U per lo
        # corrispondente valore singolare
        blocks_dct_modified = np.matmul(U * s[:, :, np.newaxis], Vh)

        # Calcola l'inversa della DCT in modalità vettoriale
        blocks_idct = idct(idct(blocks_dct_modified, axis=1, norm='ortho'), axis=2, norm='ortho')

        # Ricostruisce il frame: riportiamo i blocchi nella forma originale
        blocks_idct = blocks_idct.reshape(n_blocks_row, n_blocks_col, block, block).transpose(0, 2, 1, 3)
        frame_encoded = blocks_idct.reshape(n_blocks_row * block, n_blocks_col * block)
        # Se il frame originale era più grande (per via del taglio per multipli di block), lo copiamo
        frame[:n_blocks_row * block, :n_blocks_col * block] = frame_encoded

        return frame

    def decode_frame(self, frame, scale, scores):
        """
        Elabora il frame (matrice M x N) suddividendolo in blocchi e, in maniera vettoriale,
        estrae il bit watermark in base al primo valore singolare.
        """
        block = self._block
        rows, cols = frame.shape
        n_blocks_row = rows // block
        n_blocks_col = cols // block
        total_blocks = n_blocks_row * n_blocks_col

        # Reshape in blocchi
        blocks = frame[:n_blocks_row * block, :n_blocks_col * block].reshape(n_blocks_row, block, n_blocks_col, block)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block, block)

        # Calcola la DCT vettoriale per ciascun blocco
        blocks_dct = dct(dct(blocks, axis=1, norm='ortho'), axis=2, norm='ortho')

        # Calcola la SVD in batch
        U, s, Vh = np.linalg.svd(blocks_dct, full_matrices=False)

        # Inference del watermark: il bit è determinato dal primo valore singolare:
        # se (s0 % scale) > (scale * 0.5) allora il bit è 1, altrimenti 0.
        s0 = s[:, 0]
        inferred_bits = (np.mod(s0, scale) > scale * 0.5).astype(np.float64)

        # Raggruppa i risultati per ogni posizione del watermark (ciclo modulo _wmLen)
        for idx, bit in enumerate(inferred_bits):
            wm_index = idx % self._wmLen
            scores[wm_index].append(bit)

        return scores

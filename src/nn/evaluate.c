// Berserk is a UCI compliant chess engine written in C
// Copyright (C) 2024 Jay Honnold

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "evaluate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../bits.h"
#include "../board.h"
#include "../move.h"
#include "../movegen.h"
#include "../thread.h"
#include "../util.h"
#include "accumulator.h"

#define INCBIN_PREFIX
#define INCBIN_STYLE INCBIN_STYLE_CAMEL
#include "../incbin.h"

INCBIN(Embed, EVALFILE);

int16_t INPUT_WEIGHTS[N_FEATURES * N_HIDDEN] ALIGN;
int16_t INPUT_BIASES[N_HIDDEN] ALIGN;

int16_t OUTPUT_WEIGHTS[N_OUTPUT_BUCKETS][N_L1] ALIGN;
int16_t OUTPUT_BIAS[N_OUTPUT_BUCKETS];

inline int vecHaddEpi32(__m256i vec) {
  __m128i xmm0;
  __m128i xmm1;

  // Get the lower and upper half of the register:
  xmm0 = _mm256_castsi256_si128(vec);
  xmm1 = _mm256_extracti128_si256(vec, 1);

  // Add the lower and upper half vertically:
  xmm0 = _mm_add_epi32(xmm0, xmm1);

  // Get the upper half of the result:
  xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);

  // Add the lower and upper half vertically:
  xmm0 = _mm_add_epi32(xmm0, xmm1);

  // Shuffle the result so that the lower 32-bits are directly above the second-lower 32-bits:
  xmm1 = _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(2, 3, 0, 1));

  // Add the lower 32-bits to the second-lower 32-bits vertically:
  xmm0 = _mm_add_epi32(xmm0, xmm1);

  // Cast the result to the 32-bit integer type and return it:
  return _mm_cvtsi128_si32(xmm0);
}

int Propagate(Board* board) {

  Accumulator* acc = board->accumulators;

  const int outputBucket = (BitCount(board->occupancies[BOTH]) - 2) / 4;

  __m256i* stmAcc = (__m256i*) acc->values[board->stm];
  __m256i* oppAcc = (__m256i*) acc->values[board->xstm];

  __m256i* stmWeights = (__m256i*) & OUTPUT_WEIGHTS[outputBucket][0];
  __m256i* oppWeights = (__m256i*) & OUTPUT_WEIGHTS[outputBucket][N_HIDDEN];

  const __m256i vecZero = _mm256_setzero_si256();
  const __m256i vecQA = _mm256_set1_epi16(255);

  __m256i sum = _mm256_setzero_si256();
  __m256i v0, v1;

  for (int i = 0; i < N_HIDDEN / 16; ++i) {
    // Side to move
    v0 = _mm256_max_epi16(stmAcc[i], vecZero); // clip
    v0 = _mm256_min_epi16(v0, vecQA); // clip
    v1 = _mm256_mullo_epi16(v0, stmWeights[i]); // square
    v1 = _mm256_madd_epi16(v1, v0); // multiply with output layer
    sum = _mm256_add_epi32(sum, v1); // collect the result

    // Non side to move
    v0 = _mm256_max_epi16(oppAcc[i], vecZero);
    v0 = _mm256_min_epi16(v0, vecQA);
    v1 = _mm256_mullo_epi16(v0, oppWeights[i]);
    v1 = _mm256_madd_epi16(v1, v0);
    sum = _mm256_add_epi32(sum, v1);
  }

  int unsquared = vecHaddEpi32(sum) / 255 + OUTPUT_BIAS[outputBucket];

  return (unsquared * 550) / 16320;
}

int Predict(Board* board) {
  ResetAccumulator(board->accumulators, board, WHITE);
  ResetAccumulator(board->accumulators, board, BLACK);

  return Propagate(board);
}

const size_t NETWORK_SIZE = sizeof(int16_t) * N_FEATURES * N_HIDDEN + // input weights
                            sizeof(int16_t) * N_HIDDEN +              // input biases
                            sizeof(int16_t) * N_L1 * N_OUTPUT_BUCKETS + // output weights
                            sizeof(int16_t) * N_OUTPUT_BUCKETS;                            // output bias

INLINE void CopyData(const unsigned char* in) {
  size_t offset = 0;

  memcpy(INPUT_WEIGHTS, &in[offset], N_FEATURES * N_HIDDEN * sizeof(int16_t));
  offset += N_FEATURES * N_HIDDEN * sizeof(int16_t);
  memcpy(INPUT_BIASES, &in[offset], N_HIDDEN * sizeof(int16_t));
  offset += N_HIDDEN * sizeof(int16_t);

  int16_t BAD_OUTPUT_WEIGHTS[N_L1][N_OUTPUT_BUCKETS];

  memcpy(BAD_OUTPUT_WEIGHTS, &in[offset], N_L1 * N_OUTPUT_BUCKETS * sizeof(int16_t));
  offset += N_L1 * N_OUTPUT_BUCKETS * sizeof(int16_t);

  for (int i = 0; i < N_L1; i++) {
    for (int j = 0; j < N_OUTPUT_BUCKETS; j++) {
      OUTPUT_WEIGHTS[j][i] = BAD_OUTPUT_WEIGHTS[i][j];
    }
  }


  memcpy(OUTPUT_BIAS, &in[offset], N_OUTPUT_BUCKETS * sizeof(int16_t));

}

void LoadDefaultNN() {

  CopyData(EmbedData);
}

int LoadNetwork(char* path) {
  FILE* fin = fopen(path, "rb");
  if (fin == NULL) {
    printf("info string Unable to read file at %s\n", path);
    return 0;
  }

  uint8_t* data = malloc(NETWORK_SIZE);
  if (fread(data, sizeof(uint8_t), NETWORK_SIZE, fin) != NETWORK_SIZE) {
    printf("info string Error reading file at %s\n", path);
    return 0;
  }

  CopyData(data);

  for (int i = 0; i < Threads.count; i++)
    ResetRefreshTable(Threads.threads[i]->refreshTable);

  fclose(fin);
  free(data);

  return 1;
}

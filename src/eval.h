#ifndef EVAL_H
#define EVAL_H

#include <inttypes.h>

#include "types.h"

#define makeScore(mg, eg) ((int)((unsigned int)(eg) << 16) + (mg))
#define scoreMG(s) ((int16_t)((uint16_t)((unsigned)((s)))))
#define scoreEG(s) ((int16_t)((uint16_t)((unsigned)((s) + 0x8000) >> 16)))

extern const int PAWN_VALUE;
extern const int KNIGHT_VALUE;
extern const int BISHOP_VALUE;
extern const int ROOK_VALUE;
extern const int QUEEN_VALUE;
extern const int KING_VALUE;
extern const int MATERIAL_VALUES[];

extern int MATERIAL_AND_PSQT_VALUES[12][64];

void initPSQT();

int getPhase(Board* board);
int taper(int score, int phase);

int isMaterialDraw(Board* board);
int EvaluateSide(Board* board, int side);
int Evaluate(Board* board);

#endif
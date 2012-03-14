#ifndef LOCALCUES
#define LOCALCUES


/**
 * Performs local cue extractions.  Norientations is currently fixed at 8
 * and the multiscale computations are done at radius 5, 10, & 20
 * @param width the width of the image
 * @param height the height of the image
 * @param devL a DEVICE pointer to the image, in normalized L space [0, 1]
 * @param devA a DEVICE pointer to the image, in normalized a space [0, 1]
 * @param devB a DEVICE pointer to the image, in normalized b space [0, 1]
 * @param devTextons a DEVICE pointer to the texton labels (one label per pixel)
 * @param devBg a DEVICE pointer to the Bg cues.  They are laid out in rows like this: each row is width*height floats long, with padding to keep each row GPU aligned (making each row cuePitchInFloats long).  The rows are in this order: Scale0Orient0, Scale0Orient1, Scale0Orient2, Scale0Orient3...Scale0Orient7, Scale1Orient0, Scale1Orient1, ... Scale1Orient7, Scale2Orient0, Scale2Orient1, ... Scale2Orient7
 * @param devCga a DEVICE pointer to the Cga cues, laid out like the Bg cues.
 * @param devCgb a DEVICE pointer to the Cgb cues, laid out like the Bg cues.
 * @param devTg  a DEVICE pointer to the Tg cues, laid out like the Bg cues.
 * @param cuePitchInFloats a pointer to an integer which stores how many floats there are for each row of the cues
 */
void localCues(int width, int height, float* devL, float* devA, float* devB, int* devTextons, float** devBg, float** devCga, float** devCgb, float** devTg, int* cuePitchInFloats, int textonChoice);

#endif

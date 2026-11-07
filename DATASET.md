# Dataset provenance

Source: ML4SCI DeepLense observational images, Specific Test V in the [official instructions](https://docs.google.com/document/d/10APh49fvayGoSftzO4fGXs2HP3uvYSzG-fSrq4xHL1w/edit).

[Original archive](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view)

- Recorded size: 2,114,787,260 bytes.
- SHA-256: `ddfe91d32f79da92c5610295f3dd3475b7e40d5788e16cd866cb3e9487cb7b9c`.
- Original training: 1,730 lenses and 28,675 non-lenses.
- Original test: 195 lenses and 19,455 non-lenses.
- Three channels, 64 by 64 pixels; float32, finite values in [0,1]. Filter order is not independently established.
- Seed 429; stratified 20% validation from original training. No exact duplicates were excluded in the recorded run. Same-object independence cannot be established without coordinates.
- Values are processed intensities, not calibrated fluxes. No arc masks, PSF, or mass labels are assumed.

Download the dataset separately. Images are not included in this repository, and a code license would not grant rights to the dataset. Consult the original providers for usage terms, attribution, and required scientific citations before redistributing images or publishing a study.

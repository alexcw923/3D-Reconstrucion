
<br />
<div align="center">
  

  <h3 align="center">3D Reconstruction (Structured Light Scanning)</h3>

  
</div>

This 3D Reconstrution Tool utilizes structure light scanning data to reproduce 3D object in coordinates which are stored in .ply files after execution. Users can further utilize .ply files to reconstruct the object with different scanning view of the original object.

## Getting Started

### Calibration

We need to first calibrate the cameras with given chessboard images to store the correct settings for the left camera ```camL``` and the right camera ```camR``` in ```calibration.pickle```:

```
python calibrate.py
```

### Reconstruction

To reconstruct the object with the calibrated cameras, use the following command:

```
python reconstruct.py
```




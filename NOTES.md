# Scripts
AprilTag detector --> constructs B

Optitrack parser --> constructs A

Prepare calibration data --> synchronizes trimmed camera feeds with optitrack data; 
                            outputs A,B pose pairs for Julia solver

Run julia solver --> Reads A,B pose pairs from CSV and outputs X,Y calibration results.

run calibration -->  the Julia solver to calibrate cameras for one pose-tag pair using all available tag data.

run batch calibration --> extrinsic calibration solver using all preprocessed A,B pairs


# Important
- AprilTag outputs camera-->tag, but julia solver expects tag-->camera

- Optitrack gives world-->rig, but equation needs rig-->world

- Optitrack and OpenCV coordinate systems aren't aligned, so both A and B are misdefined, breaking the transformation chain.
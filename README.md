## Super resolution algorithms for optical payloads in Earth Observation missions

### Explanation
Source code and documentation for my master's thesis on super-resolution algorithms for optical payloads in Earth observation missions. The proposed approach involves two main methodologies: the first one consists of selecting images obtained by shifting pixels using a specific registration algorithm and iteratively interpolating non-uniformly spaced samples based on gray pixel correction (algorithm1). The second approach employs image registration with sub-pixel shifting and interpolation based on curvelets (algorithm2). "Matrix" refers to the second algorithm applied to numerical matrices. It is used for testing purposes.

### Installation
To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone linkRepository
    ```

2. Run the requirements file to set up the working environment:

    ```bash
    pip install -r requirements.txt
    ```

### Curvelops
Curvelops is a Python wrapper for CurveLab's 2D and 3D curvelet transforms. It uses the PyLops design framework to provide forward and inverse curvelet transforms as matrix-free linear operations. If you are still confused, take a look at some examples below or visit the PyLops website!

If you encounter issues via the requirements.txt file, follow the installation guide used for the Curvelops library available here:

[Curvelops Installation Guide](https://github.com/PyLops/curvelops?tab=readme-ov-file)

### Citation
If you want to cite this project, you can use the following BibTeX entry:
```
@misc{github.nome,
    title = {{Nome}},
    howpublished = {\url{https:...}},
    author = {Camilla De Martino},
    year = {2024},
    description = {}
}
```
### "Future implementations"
Apply the Daubechies filter to the second algorithm to attenuate interpolation errors along the edges.

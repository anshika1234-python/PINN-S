# Physics-Informed Neural Network (PINN) for Maxwell's Equations

![EM Wave Animation](em_wave_animation.gif)

## 📝 Project Overview

This project showcases a Physics-Informed Neural Network (PINN) built from scratch in PyTorch to solve a fundamental problem in electromagnetism: the propagation of a 1D wave as described by **Maxwell's equations**.

Unlike traditional deep learning models that require large datasets, this PINN learns the underlying physics directly. The governing partial differential equations are embedded into the loss function, compelling the model to find a solution that is not only data-consistent but also physically valid. This approach is a powerful demonstration of how deep learning can be applied to solve complex scientific and engineering problems from first principles.

This project was developed as a capstone piece to bridge my academic background in **MSc Physics** with practical, industry-relevant skills in **AI and Machine Learning**.

## 🧠 The Challenge: Overcoming "Lazy" Neural Networks

Training PINNs is notoriously difficult. A standard neural network, when faced with a complex physical system, will often converge to a "trivial" or physically incorrect solution (like a static or zero-field) because it represents an easier local minimum for the optimizer.

This project successfully overcame these challenges through advanced techniques:
1.  **Fourier Feature Mapping:** To combat the network's inherent "spectral bias" (difficulty learning high-frequency functions), the inputs `(t, x)` were transformed into a high-dimensional feature space, making it much easier for the model to learn the complex dynamics of wave propagation.
2.  **Two-Phase "Annealing" Training:** A sophisticated training strategy was implemented:
    * **Phase 1:** The model was first trained *only* on the initial and boundary conditions with a high learning rate, forcing it to perfectly learn the starting shape of the wave.
    * **Phase 2:** The physics loss was "turned on" with a high weight, and the learning rate was lowered. This fine-tuned the model, forcing it to learn how the initial shape should evolve over time according to Maxwell's equations.

## 📈 Results

The final trained model successfully learned the complex dynamics of a propagating electromagnetic wave. The animation above shows the initial Gaussian pulse correctly splitting into two waves traveling in opposite directions, demonstrating that the model has learned the solution to the governing PDEs.

## 🛠️ Technology Stack

-   **Language:** Python
-   **Core Libraries:** PyTorch, NumPy
-   **Visualization:** Matplotlib

## 🚀 How to Run

This project was developed in a VS Code Codespace and can be run in any environment with Python and the required libraries.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/anshika1234-python/PINN-S.git](https://github.com/anshika1234-python/PINN-S.git)
    cd PINN-S
    ```

2.  **Install the required libraries:**
    ```bash
    pip install torch numpy matplotlib
    ```

3.  **Run the main script:**
    ```bash
    python main.py
    ```
    The script will initiate the two-phase training process. Upon completion, it will save the results as `em_wave_animation.gif` in the project directory. For faster training, a GPU-enabled environment (like Google Colab) is recommended.

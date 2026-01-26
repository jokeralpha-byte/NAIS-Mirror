# NAIS-Mirror

A Demonstration of Real-Time 360-Degree Video Streaming

## Overview

NAIS-Mirror is a proof-of-concept implementation showcasing an innovative method for real-time 360-degree video streaming. This approach enables seamless, immersive video experiences by efficiently processing and transmitting panoramic content.

## Framework of Mirror360

The following diagram illustrates the core architecture of Mirror360, highlighting key components such as data ingestion, processing pipelines, and output rendering.

![Mirror360 Framework](images/framework.png)
## Visual quality demonstration of the method
This is a comparison of the visual quality of our method with other methods, and a comparison of the CDF visual quality elliptic curve.

![Output Mask-Based Visualization](images/视觉质量效果图.gif)

The following are the visual quality performance results of various baselines for our method.

![Output Mask-Based Visualization](images/展示图.gif)
## Reference Result in a Visualization Scenario

Below is a GIF demonstrating the method's output in a practical visualization use case. It showcases the real-time masking and rendering capabilities, ensuring smooth performance even in dynamic environments.

![Output Mask-Based Visualization](images/output_mask_based.gif)



## Getting Started

To set up and run the demonstration:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/NAIS-Mirror.git
   ```

2. Install dependencies (assuming Python-based; adjust as needed):
   ```
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```
   python main_demo.py
   ```

For detailed instructions, configuration options, and troubleshooting, refer to the [documentation](docs/README.md).

## Contributions

We welcome contributions! Please fork the repository and submit pull requests for bug fixes, enhancements, or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

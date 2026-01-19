# Waste Classification Service

Waste Classification Service is an application based on Machine Learning models to classify waste from an image containing a waste object.

The application was developed following deep learning techniques including image-based waste training and evaluation, which can serve as the basis for automated recycling and smart waste management systems.

For further information about the project, see [Documentation](docs/project-documentation.md)

### How the app works
![alt text](docs/assets/app-diagram.png)

## How to run

### Prerequisites
- Python 3.11+
- uv (to create and manage the virtual environment)
- (Optional) Docker if you prefer containerized execution

### Clone the repository
 
```bash
git clone https://github.com/alexispdDev/waste-classification
cd waste-classification
```

### A. Build and Run the Application Using Docker

From the root directory of the project, build the image using:
```bash
docker build -t waste-classification:v1 .
docker run -it --rm -p 8080:8080 waste-classification:v1
```

### B. Locally

### Set Up the Environment
```bash
uv venv
source .venv/bin/activate      # On Linux/Mac
# or
.venv\Scripts\activate         # On Windows
```

### Install package and dependencies
```bash
uv sync
uv pip install -e .
```

### Run the Project

To run the API service:   
```bash
python -m waste_classification.serve
```

### Train a model

If you want to train a model, you would need to first download the data. This repository includes a convenience script for downloading the data:
```bash
./bin/download-dataset.sh
```
Afterwards, you can run the training script, by calling the following command:
```bash
python -m waste_classification.train
```
This repository already includes a pre-computed model that you can use to generate images without training, but the commands are safe enough to generate new models.

## How to make a classification

Run the application as described in the "How to run" section via Docker or locally.  

### A. Via curl

1. Open a terminal.

2. Input this example request:
```bash
curl -X 'POST' \
  'http://localhost:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://storage.googleapis.com/kagglesdsdata/datasets/9083965/14238273/Garbage%20classification/test/cardboard/cardboard112.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260119%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260119T094428Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6efdbb784d786342ee5e566d4562048552db1f0e866da3178949c3064f5cad5a6bb69da0ec9bc5e2a07518269d78ca32d02c3543f76dbc101bd87f0ecbefc6f4e76fa83ca75f325c21061b967902e83a3f273917ed7ac901768d15f2a6118d8817421b8c24d33c7363b161bb8df86b697fc3f3572e813f771561b1fe0da62d36da59e7dad2a4d469f7bcaea3998758ad43530360047a3186bfae2dc04a2ccd6079b2639752f8d1e3152c952b19858e0b34ffbaa0c0bd9828ca2b55dccd44b1fc11adfa72e351acac974bad8d13a51ace5e89c6d43247960041a067cf048d13ae0507188c25b39bc9bfa3fac59ade3383782374cb6d58ae53140361ce96256583"
}'
```
3. The score classification appears in console as follow:
```bash
{"predictions":{"cardboard":10.491369247436523,"glass":-5.355796813964844,"metal":-5.617727756500244,"paper":-3.2971396446228027,"plastic":-8.81702995300293,"trash":-5.612159252166748},"top_class":"cardboard","top_probability":10.491369247436523}
```
### B. Via Browser

1. Open a browser and access to: http://localhost:8080/docs
![Score Prediction docs](docs/assets/localhost-docs-1.png)

2. Drop down the green box with the POST label.
![Drop down green box](docs/assets/localhost-docs-drop-down-box.png)

3. Click on "Try it out" button.
![Click Try it out](docs/assets/localhost-docs-click-try-out.png)

4. Replace the json object in the "Edit Value | Schema" box for:   
```
{
    "url": "https://storage.googleapis.com/kagglesdsdata/datasets/9083965/14238273/Garbage%20classification/test/cardboard/cardboard112.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260119%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260119T094428Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6efdbb784d786342ee5e566d4562048552db1f0e866da3178949c3064f5cad5a6bb69da0ec9bc5e2a07518269d78ca32d02c3543f76dbc101bd87f0ecbefc6f4e76fa83ca75f325c21061b967902e83a3f273917ed7ac901768d15f2a6118d8817421b8c24d33c7363b161bb8df86b697fc3f3572e813f771561b1fe0da62d36da59e7dad2a4d469f7bcaea3998758ad43530360047a3186bfae2dc04a2ccd6079b2639752f8d1e3152c952b19858e0b34ffbaa0c0bd9828ca2b55dccd44b1fc11adfa72e351acac974bad8d13a51ace5e89c6d43247960041a067cf048d13ae0507188c25b39bc9bfa3fac59ade3383782374cb6d58ae53140361ce96256583"
}
```
![Replace json](docs/assets/replace-json.png)
5. Click on Execute.

6. The classified waste is shown below.
![Score prediction](docs/assets/server-response.png)

## Kubernetes Deployments

### Pre-requisites

* A Kubernetes cluster.
* `kubectl` with a compatible version for your cluster.
* Your user must have enough rights to create the following:
    - Namespace
    - Deployments
    - Services
    - Horizontal pod auto-scales

> [!IMPORTANT]
> If using Kind, you would need to load your image to the cluster (folow the instructions [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/10-kubernetes/workshop#loading-image-to-kind), but remember to use the correct name (in this case, `waste-classification:v1`)).
> 
> If you're not using Kind, then you're responsible to make the image available to the cluster. You would need to load the image into a separate image registry, or manually load the image into the cluster's nodes.

> [!IMPORTANT]
> If using Kind, please ensure that you have both installed the metrics server and applied the patch to get it working. Follow the instructions [in this link](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/10-kubernetes/workshop#horizontal-pod-autoscaling).

### Deploy

```bash
kubectl apply -f k8s/manifest.yaml
```

### How to use it

For Kind users, the node port won't be available, so you would need to port-forward the service into your machine, so you can use it:
```bash
$ kubectl port-forward -n waste-classification service/waste-classification 30080:8080
```
and, afterwards, you can interact with your deployment at http://localhost:30080.

If you're using other clusters, the service is exposed as a NodePort to port 30080, so you can interact with this service through http://NODE_IP:30080.

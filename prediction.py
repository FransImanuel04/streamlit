{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0a6beefa-4b86-4d66-ab34-cb675afc0d24",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "def predict(data):\n",
    "    clf = joblib.load(\"rf_model.sav\")\n",
    "    return clf.predict(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7cfbaf46-51d3-49c7-9a07-34697b05f98a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

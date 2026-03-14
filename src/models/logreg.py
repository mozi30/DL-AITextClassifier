import numpy as np

# gpt
class SoftmaxLogReg:
    """
    Multiclass Logistic Regression using Softmax.

    This model learns a linear classifier that maps input features
    (e.g., TF-IDF vectors) to probabilities over multiple classes.

    Mathematical model:

        logits = X @ W + b
        probabilities = softmax(logits)

    where:

        X : input features matrix          (N x D)
        W : weight matrix                  (D x C)
        b : bias vector                    (C)
        C : number of classes
        N : number of samples
        D : number of features

    The model is trained using:
        - Cross-Entropy Loss
        - Mini-Batch Gradient Descent
        - L2 Regularization
    """

    def __init__(self, input_dim, num_classes, lr=0.1, reg=1e-4, seed=42):
        """
        Initialize the model.

        Parameters
        ----------
        input_dim : int
            Number of input features (e.g., TF-IDF dimension)

        num_classes : int
            Number of output classes

        lr : float
            Learning rate used in gradient descent

        reg : float
            L2 regularization strength

        seed : int
            Random seed for reproducibility
        """

        # Random number generator
        rng = np.random.default_rng(seed)

        # Weight matrix
        # Shape: (input_dim, num_classes)
        # Each column corresponds to a class
        self.W = rng.normal(0, 0.01, (input_dim, num_classes))

        # Bias vector
        # Shape: (num_classes,)
        self.b = np.zeros(num_classes)

        # Training hyperparameters
        self.lr = lr
        self.reg = reg
        self.num_classes = num_classes


    def _softmax(self, logits):
        """
        Compute softmax probabilities.

        Softmax converts raw scores (logits) into probabilities.

        Formula:

            softmax(z_i) = exp(z_i) / sum_j exp(z_j)

        Parameters
        ----------
        logits : ndarray (batch_size x num_classes)

        Returns
        -------
        probabilities : ndarray (batch_size x num_classes)
        """

        # Numerical stability trick:
        # subtract the maximum value so exp() does not overflow
        logits = logits - np.max(logits, axis=1, keepdims=True)

        exp = np.exp(logits)

        # Normalize so rows sum to 1
        return exp / np.sum(exp, axis=1, keepdims=True)


    def forward(self, X):
        """
        Forward pass of the model.

        Computes class scores and probabilities.

        Parameters
        ----------
        X : ndarray (batch_size x input_dim)

        Returns
        -------
        logits : ndarray (batch_size x num_classes)
            Raw class scores

        probs : ndarray (batch_size x num_classes)
            Softmax probabilities
        """

        # Linear classifier
        logits = X @ self.W + self.b

        # Convert scores → probabilities
        probs = self._softmax(logits)

        return logits, probs


    def compute_loss(self, X, y):
        """
        Compute training loss.

        Loss = Cross-Entropy Loss + L2 Regularization

        Cross-Entropy measures how well predicted probabilities
        match the true labels.

        Parameters
        ----------
        X : ndarray (N x D)
        y : ndarray (N)

        Returns
        -------
        loss : float
        """

        N = X.shape[0]

        logits, probs = self.forward(X)

        # Select the probability of the correct class
        # probs[i, y[i]]
        correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-12)

        # Average cross-entropy loss
        data_loss = np.mean(correct_log_probs)

        # L2 regularization penalty
        reg_loss = self.reg * np.sum(self.W * self.W)

        return data_loss + reg_loss


    def predict(self, X):
        """
        Predict class labels.

        Returns the class with the highest probability.

        Parameters
        ----------
        X : ndarray (N x D)

        Returns
        -------
        predictions : ndarray (N)
        """

        _, probs = self.forward(X)

        # Choose the class with highest probability
        return np.argmax(probs, axis=1)


    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=30,
        batch_size=256,
    ):
        """
        Train the model using Mini-Batch Gradient Descent.

        Training loop:
            1. Shuffle data
            2. Split into mini-batches
            3. Compute predictions
            4. Compute gradients
            5. Update parameters

        Parameters
        ----------
        X_train : ndarray (N x D)
        y_train : ndarray (N)

        X_val : ndarray
            Optional validation set

        y_val : ndarray

        epochs : int
            Number of passes through the dataset

        batch_size : int
            Size of mini-batches used for training
        """

        N = X_train.shape[0]

        for epoch in range(epochs):

            # Shuffle dataset each epoch
            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Mini-batch loop
            for start in range(0, N, batch_size):

                end = start + batch_size

                # Mini-batch
                Xb = X_train[start:end]
                yb = y_train[start:end]

                B = Xb.shape[0]

                # Forward pass
                logits, probs = self.forward(Xb)

                # Convert labels → one-hot encoding
                Y = np.zeros((B, self.num_classes))
                Y[np.arange(B), yb] = 1

                # Gradient of cross-entropy + softmax
                dZ = (probs - Y) / B

                # Gradient w.r.t weights
                dW = Xb.T @ dZ + 2 * self.reg * self.W

                # Gradient w.r.t bias
                db = np.sum(dZ, axis=0)

                # Gradient descent update
                self.W -= self.lr * dW
                self.b -= self.lr * db

            # Compute metrics after each epoch
            train_loss = self.compute_loss(X_train, y_train)
            train_acc = np.mean(self.predict(X_train) == y_train)

            if X_val is not None:

                val_loss = self.compute_loss(X_val, y_val)
                val_acc = np.mean(self.predict(X_val) == y_val)

                print(
                    f"Epoch {epoch+1:03d} | "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f}"
                )

            else:

                print(
                    f"Epoch {epoch+1:03d} | "
                    f"train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f}"
                )
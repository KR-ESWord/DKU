# DKU
## AI 특론
- revision에서 수정된 사항
  - 기존의 Group Batch Normalization이 아닌 Ghost Batch Normalization 코드 작성(Tabnet_TF_2의 11번, 12번 block)
   - 수정 이유 : 기존에는 논문에서 사용한 Ghost Batch Normalization이 아닌 많이 사용되었던 Group Batch Normalization을 이용하여 구성을 하였지만,
              이번에 논문의 알고리즘을 그대로 적용해보고자 pytorch 코드를 tensorflow 코드로 구현
   - 수정 결과 : Tensorflow에서 아직은 Ghost Batch Normalization의 내부 연산으 일부 기능이 지원되지 않아서 모델 학습 시 에러 발생

### 모형 설명
- TabNet
  -  Deep Neural Network와 DecisionTree-based 모델의 장점을 계승한 정형 데이터 분석을 위한 딥러닝 모형
  -  기존 기계 학습 모델보다 우수한 성능을 보인다고 하기에는 어려움이 있지만, 성능보다는 딥러닝의 장점이 필요할 때 주로 사용
  -  특징
    1) 전처리가 거의 필요 없고 최적화 알고리즘으로 경사하강법을 사용하는 구조로 end-to-end 학습에 유연하게 적용 가능
    2) Sequential attention을 사용하여 feature selection의 이유 추적 가능 -> interpretability 확보(local & global)
  
  <img src=https://user-images.githubusercontent.com/59715960/234817143-c58d5125-1f07-49a5-af9d-1805c03a20ea.png />
  
  - Encoder  
<img src=https://user-images.githubusercontent.com/59715960/234817915-8102e9be-7526-4f6c-8a11-807eb9ec40c5.png width="600" height="300"/>
  
   - Step의 여러 단계를 거치면서 각 Step 마다 feature selection mask를 구하여 각 Step 별 핵심 feature 파악 가능(local)
   - 모든 Step의 mask를 합하여 입력받은 전체 데이터에 대해서도 feature의 중요도 파악 가능(global)
  
  - Result
    <img src=https://user-images.githubusercontent.com/59715960/235048302-64b58d87-aabb-4ac0-a349-17ff95f7c836.png width="400" height="300"/> 

### 요구사항
- Python version 3.7.x, 3.8.x, 3.9.x, 3.10.x
- Tensorflow 2.x
(pytorch-tabnet 라이브러리로 구현이 되었지만, tensorflow를 사용하여 implementation 수행)

### 실행 방법
  1. Import Libraries
<pre>
<code>
import os
import shutil
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
</code>
</pre>

  2. Data Load
<pre>
<code>
train_size = 125
BATCH_SIZE = 50

def transform(ds):
    features = tf.unstack(ds['features'])
    labels = ds['label']

    x = dict(zip(col_names, features))
    y = tf.one_hot(labels, 3)
    return x, y

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
ds_full = tfds.load(name="iris", split=tfds.Split.TRAIN)
ds_full = ds_full.shuffle(150, seed=0)
</code>
</pre>

  3. Pre Processing
<pre>
<code>
feature_columns = []
for col_name in col_names:
    feature_columns.append(tf.feature_column.numeric_column(col_name))
</code>
</pre>

  4. Dataset Split
<pre>
<code>
ds_train = ds_full.take(train_size)
ds_train = ds_train.map(transform)
ds_train = ds_train.batch(BATCH_SIZE)

ds_test = ds_full.skip(train_size)
ds_test = ds_test.map(transform)
ds_test = ds_test.batch(BATCH_SIZE)
</code>
</pre>

  5. Define Model
<pre>
<code>
class TransformBlock(tf.keras.Model):

    def __init__(self, features,
                 norm_type,
                 momentum=0.9,
                 virtual_batch_size=None,
                 groups=2,
                 block_name='',
                 **kwargs):
        super(TransformBlock, self).__init__(**kwargs)

        self.features = features
        self.norm_type = norm_type
        self.momentum = momentum
        self.groups = groups
        self.virtual_batch_size = virtual_batch_size

        self.transform = tf.keras.layers.Dense(self.features, use_bias=False, name=f'transformblock_dense_{block_name}')

        if norm_type == 'batch':
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                         virtual_batch_size=virtual_batch_size,
                                                         name=f'transformblock_bn_{block_name}')
        elif norm_type == "ghost":
            self.bn = GhostBatchNormalization(virtual_divider=virtual_batch_size, momentum=momentum)

        else:
            self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')

    def call(self, inputs, training=None):
        x = self.transform(inputs)
        x = self.bn(x, training=training)
        return x


class TabNet(tf.keras.Model):

    def __init__(self, feature_columns,
                 feature_dim=64,
                 output_dim=64,
                 num_features=None,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=2,
                 epsilon=1e-5,
                 random_state = None,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'batch' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNet, self).__init__(**kwargs)
        if random_state is not None:
            tf.keras.utils.set_random_seed(random_state)

        # Input checks
        if feature_columns is not None:
            if type(feature_columns) not in (list, tuple):
                raise ValueError("`feature_columns` must be a list or a tuple.")

            if len(feature_columns) == 0:
                raise ValueError("`feature_columns` must be contain at least 1 tf.feature_column !")

            if num_features is None:
                num_features = len(feature_columns)
            else:
                num_features = int(num_features)

        else:
            if num_features is None:
                raise ValueError("If `feature_columns` is None, then `num_features` cannot be None.")

        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        if feature_dim <= output_dim:
            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        num_groups = max(1, int(num_groups))
        epsilon = float(epsilon)

        if relaxation_factor < 0.:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        if norm_type not in ['batch', 'group']:
            raise ValueError("`norm_type` must be either `batch` or `group`")

        self.feature_columns = feature_columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.norm_type = norm_type
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)

            if self.norm_type == 'batch':
                self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')
            else:
                self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')

        else:
            self.input_features = None
            self.input_bn = None

        self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f1')

        self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f2')

        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f3_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f4_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_coef_list = [
            TransformBlock(self.num_features, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'coef_{i}')
            for i in range(self.num_decision_steps - 1)
        ]

        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)

        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.

        for ni in range(self.num_decision_steps):
            # Feature transformer with two shared and two decision step dependent
            # blocks is used below.=
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)

            if (ni > 0 or self.num_decision_steps == 1):
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim:]

            if ni < (self.num_decision_steps - 1):
                # Determines the feature masks via linear and nonlinear
                # transformations, taking into account of aggregated feature use.
                mask_values = self.transform_coef_list[ni](features_for_coef, training=training)
                mask_values *= complementary_aggregated_mask_values
                mask_values = sparsemax(mask_values, axis=-1)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complementary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (
                                     tf.cast(self.num_decision_steps - 1, tf.float32))

                # Add entropy loss
                entropy_loss = total_entropy

                # Feature selection.
                masked_features = tf.multiply(mask_values, features)

                # Visualization of the feature selection mask at decision step ni
                # tf.summary.image(
                #     "Mask for step" + str(ni),
                #     tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                #     max_outputs=1)
                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                # This branch is needed for correct compilation by tf.autograph
                entropy_loss = 0.

        # Adds the loss automatically
        self.add_loss(self.sparsity_coefficient * entropy_loss)

        # Visualization of the aggregated feature importances
        # tf.summary.image(
        #     "Aggregated mask",
        #     tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
        #     max_outputs=1)

        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask

        return output_aggregated

    @property
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask


class TabNetClassifier(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_classes,
                 num_features=None,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 random_state = None,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_classes: Number of classes.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'group' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             random_state=random_state,
                             **kwargs)

        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.clf(self.activations)

        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)


class TabNetRegressor(tf.keras.Model):

    def __init__(self, feature_columns,
                 num_regressors,
                 num_features=None,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 random_state = None,
                 **kwargs):
        """
        Tensorflow 2.0 implementation of [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)

        # Hyper Parameter Tuning (Excerpt from the paper)
        We consider datasets ranging from ∼10K to ∼10M training points, with varying degrees of fitting
        difficulty. TabNet obtains high performance for all with a few general principles on hyperparameter
        selection:

            - Most datasets yield the best results for Nsteps ∈ [3, 10]. Typically, larger datasets and
            more complex tasks require a larger Nsteps. A very high value of Nsteps may suffer from
            overfitting and yield poor generalization.

            - Adjustment of the values of Nd and Na is the most efficient way of obtaining a trade-off
            between performance and complexity. Nd = Na is a reasonable choice for most datasets. A
            very high value of Nd and Na may suffer from overfitting and yield poor generalization.

            - An optimal choice of γ can have a major role on the overall performance. Typically a larger
            Nsteps value favors for a larger γ.

            - A large batch size is beneficial for performance - if the memory constraints permit, as large
            as 1-10 % of the total training dataset size is suggested. The virtual batch size is typically
            much smaller than the batch size.

            - Initially large learning rate is important, which should be gradually decayed until convergence.

        Args:
            feature_columns: The Tensorflow feature columns for the dataset.
            num_regressors: Number of regression variables.
            feature_dim (N_a): Dimensionality of the hidden representation in feature
                transformation block. Each layer first maps the representation to a
                2*feature_dim-dimensional output and half of it is used to determine the
                nonlinearity of the GLU activation where the other half is used as an
                input to GLU, and eventually feature_dim-dimensional output is
                transferred to the next layer.
            output_dim (N_d): Dimensionality of the outputs of each decision step, which is
                later mapped to the final classification or regression output.
            num_features: The number of input features (i.e the number of columns for
                tabular data assuming each feature is represented with 1 dimension).
            num_decision_steps(N_steps): Number of sequential decision steps.
            relaxation_factor (gamma): Relaxation factor that promotes the reuse of each
                feature at different decision steps. When it is 1, a feature is enforced
                to be used only at one decision step and as it increases, more
                flexibility is provided to use a feature at multiple decision steps.
            sparsity_coefficient (lambda_sparse): Strength of the sparsity regularization.
                Sparsity may provide a favorable inductive bias for convergence to
                higher accuracy for some datasets where most of the input features are redundant.
            norm_type: Type of normalization to perform for the model. Can be either
                'group' or 'group'. 'group' is the default.
            batch_momentum: Momentum in ghost batch normalization.
            virtual_batch_size: Virtual batch size in ghost batch normalization. The
                overall batch size should be an integer multiple of virtual_batch_size.
            num_groups: Number of groups used for group normalization.
            epsilon: A small number for numerical stability of the entropy calculations.
        """
        super(TabNetRegressor, self).__init__(**kwargs)

        self.num_regressors = num_regressors

        self.tabnet = TabNet(feature_columns=feature_columns,
                             num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             random_state=random_state,
                             **kwargs)

        self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False, name='regressor')

    def call(self, inputs, training=None):
        self.activations = self.tabnet(inputs, training=training)
        out = self.regressor(self.activations)
        return out

    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)
        
# Group Norm does better for small datasets
model = TabNetClassifier(feature_columns, num_classes=3,
                        feature_dim=8, output_dim=4,
                        num_decision_steps=4, relaxation_factor=1.0,
                        sparsity_coefficient=1e-5, batch_momentum=0.98,
                        virtual_batch_size=None, norm_type='ghost',
                        num_groups=1)

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=100, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.legacy.Adam(lr)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
</code>
</pre>

  6. Model Training
<pre>
<code>
model.fit(ds_train, epochs=100, validation_data=ds_test, verbose=2)
</code>
</pre>

  7. Predict
<pre>
<code>
# Prediction
pred = model.predict(ds_test)
</code>
</pre>

### pseudocode
  - Sparse max
  
    <img width="300" alt="스크린샷 2023-05-12 오후 1 46 21" src="https://github.com/KR-ESWord/DKU/assets/59715960/dc63986d-0f35-4c96-969b-5811484d81f0">
    
    <pre>
    <code>
    def sparsemax(z):
      sum_all_z = sum(z)
      z_sorted = sorted(z, reverse=True)
      k = np.arange(len(z))
      k_array = 1 + k * z_sorted
      z_cumsum = np.cumsum(z_sorted) - z_sorted
      k_selected = k_array > z_cumsum
      k_max = np.where(k_selected)[0].max() + 1
      threshold = (z_cumsum[k_max-1] - 1) / k_max
      return np.maximum(z-threshold, 0)
    </code>
    </pre>

### To Do
  - Visualize Mask

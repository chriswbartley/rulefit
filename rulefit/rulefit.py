"""Linear model of tree-based decision rules

This method implement the RuleFit algorithm

The module structure is the following:

- ``RuleCondition`` implements a binary feature transformation
- ``Rule`` implements a Rule composed of ``RuleConditions``
- ``RuleEnsemble`` implements an ensemble of ``Rules``
- ``RuleFit`` implements the RuleFit algorithm

"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import train_test_split,StratifiedKFold
from functools import reduce
import  scipy 
import cvglmnet
import cvglmnetCoef
import cvglmnetPredict
from monoboost import MonoBoost,MonoLearner,MonoBoostEnsemble

class GLMCV():
    def __init__(self,family='gaussian',incr_feats=None,decr_feats=None):
        self.cvfit=None
        self.family=family
        self.incr_feats=np.asarray([]) if incr_feats is None else np.asarray(incr_feats)
        self.decr_feats=np.asarray([]) if decr_feats is None else np.asarray(decr_feats)

    def fit(self,x,y):        
        cv_loss='class' if self.family=='binomial' else 'deviance'
        coef_limits=scipy.array([[scipy.float64(-scipy.inf)], [scipy.float64(scipy.inf)]]) # default, no limits on coefs
        # set up constraints
        if  len( self.incr_feats)>0 or  len(self.decr_feats)>0 :
            coef_limits=np.zeros([2,x.shape[1]])
            for i_feat in np.arange(x.shape[1]):
                coef_limits[0,i_feat]=-np.inf if i_feat not in self.incr_feats-1 else 0.
                coef_limits[1,i_feat]=np.inf if i_feat not in self.decr_feats-1 else 0.
            coef_limits=scipy.array(coef_limits)
        self.cvfit = cvglmnet.cvglmnet(x = x.copy(), y = y.copy(), nfolds=5,family = self.family, ptype = cv_loss, nlambda = 20,intr=True,cl=coef_limits)
        coef=cvglmnetCoef.cvglmnetCoef(self.cvfit, s = 'lambda_min')
        self.coef_=coef[1:,0]
        self.intercept_=coef[0,0]
    def predict(self,x):
        return cvglmnetPredict.cvglmnetPredict(self.cvfit, newx = x, s = 'lambda_min', ptype = 'class')

    def predict_proba(self,x):
        return cvglmnetPredict.cvglmnetPredict(self.cvfit, newx = x, s = 'lambda_min', ptype = 'link')
        
   

class RuleCondition():
    """Class for binary rule condition

    Warning: this class should not be used directly.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 operator,
                 support,
                 feature_name = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.operator = operator
        self.support = support
        self.feature_name = feature_name


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.feature_name:
            feature = self.feature_name
        else:
            feature = self.feature_index
        return "%s %s %s" % (feature, self.operator, self.threshold)

    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        if self.operator == "<=":
            res =  1 * (X[:,self.feature_index] <= self.threshold)
        elif self.operator == ">":
            res = 1 * (X[:,self.feature_index] > self.threshold)
        return res

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.feature_index, self.threshold, self.operator, self.feature_name))


class FriedScale():
    """Performs scaling of linear variables according to Friedman et al. 2005 Sec 5

    Each variable is firsst Winsorized l->l*, then standardised as 0.4 x l* / std(l*)
    Warning: this class should not be used directly.
    """    
    def __init__(self,trim_quantile=0.0):
        self.trim_quantile=trim_quantile
        self.scale_multipliers=None
        self.winsor_lims=None
        
    def train(self,X):
        # get winsor limits
        self.winsor_lims=np.ones([2,X.shape[1]])*np.inf
        self.winsor_lims[0,:]=-np.inf
        if self.trim_quantile>0:
            for i_col in np.arange(X.shape[1]):
                lower=np.percentile(X[:,i_col],self.trim_quantile*100)
                upper=np.percentile(X[:,i_col],100-self.trim_quantile*100)
                self.winsor_lims[:,i_col]=[lower,upper]
        # get multipliers
        scale_multipliers=np.ones(X.shape[1])
        for i_col in np.arange(X.shape[1]):
            num_uniq_vals=len(np.unique(X[:,i_col]))
            if num_uniq_vals>2: # don't scale binary variables which are effectively already rules
                X_col_winsorised=X[:,i_col].copy()
                X_col_winsorised[X_col_winsorised<self.winsor_lims[0,i_col]]=self.winsor_lims[0,i_col]
                X_col_winsorised[X_col_winsorised>self.winsor_lims[1,i_col]]=self.winsor_lims[1,i_col]
                scale_multipliers[i_col]=0.4/np.std(X_col_winsorised)
        self.scale_multipliers=scale_multipliers
        
    def scale(self,X):
        return X*self.scale_multipliers

class Rule():
    """Class for binary Rules from list of conditions

    Warning: this class should not be used directly.
    """
    def __init__(self,
                 rule_conditions,value):
        self.conditions = set(rule_conditions)
        self.support = min([x.support for x in rule_conditions])
        self.value=value
        self.rule_direction=None
    def transform(self, X):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, 1)
        """
        rule_applies = [condition.transform(X) for condition in self.conditions]
        return reduce(lambda x,y: x * y, rule_applies)

    def __str__(self):
        return  " & ".join([x.__str__() for x in self.conditions])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return sum([condition.__hash__() for condition in self.conditions])

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def extract_rules_from_tree(tree, feature_names=None):
    """Helper to turn a tree into as set of rules
    """
    rules = set()

    def traverse_nodes(node_id=0,
                       operator=None,
                       threshold=None,
                       feature=None,
                       conditions=[]):
        if node_id != 0:
            if feature_names is not None:
                feature_name = feature_names[feature]
            else:
                feature_name = feature
            rule_condition = RuleCondition(feature_index=feature,
                                           threshold=threshold,
                                           operator=operator,
                                           support = tree.n_node_samples[node_id] / float(tree.n_node_samples[0]),
                                           feature_name=feature_name)
            new_conditions = conditions + [rule_condition]
            #new_rule = Rule(new_conditions,tree.value[node_id][0][0])
            #rules.update([new_rule])
        else:
            new_conditions = []
                ## if not terminal node
        if tree.children_left[node_id] != tree.children_right[node_id]: #not tree.feature[node_id] == -2:
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            
            left_node_id = tree.children_left[node_id]
            traverse_nodes(left_node_id, "<=", threshold, feature, new_conditions)
            
            right_node_id = tree.children_right[node_id]
            traverse_nodes(right_node_id, ">", threshold, feature, new_conditions)
        else: # a leaf node
            if len(new_conditions)>0:
                new_rule = Rule(new_conditions,tree.value[node_id][0][0])
                rules.update([new_rule])
            else:
                print('********** WHAT THE??? ****** ' + str(tree.node_count))
            return None

    traverse_nodes()
    
    return rules



class RuleEnsemble():
    """Ensemble of binary decision rules

    This class implements an ensemble of decision rules that extracts rules from
    an ensemble of decision trees.

    Parameters
    ----------
    tree_list: List or array of DecisionTreeClassifier or DecisionTreeRegressor
        Trees from which the rules are created

    feature_names: List of strings, optional (default=None)
        Names of the features

    Attributes
    ----------
    rules: List of Rule
        The ensemble of rules extracted from the trees
    """
    def __init__(self,
                 tree_list,
                 feature_names=None):
        self.tree_list = tree_list
        self.feature_names = feature_names
        self.rules = set()
        ## TODO: Move this out of __init__
        self._extract_rules()
        self.rules=list(self.rules)

    def _extract_rules(self):
        """Recursively extract rules from each tree in the ensemble

        """
        for tree in self.tree_list:
            rules = extract_rules_from_tree(tree[0].tree_,feature_names=self.feature_names)
            self.rules.update(rules)

    def filter_rules(self, func):
        self.rules = filter(lambda x: func(x), self.rules)

    def filter_short_rules(self, k):
        self.filter_rules(lambda x: len(x.conditions) > k)

    def transform(self, X,coefs=None):
        """Transform dataset.

        Parameters
        ----------
        X: array-like matrix, shape=(n_samples, n_features)

        Returns
        -------
        X_transformed: array-like matrix, shape=(n_samples, n_out)
            Transformed dataset. Each column represents one rule.
        """
        rule_list=list(self.rules) 
        if   coefs is None:
            return np.array([rule.transform(X) for rule in rule_list]).T
        else: # else use the coefs to filter the rules we bother to interpret
            res= np.array([rule_list[i_rule].transform(X) for i_rule in np.arange(len(rule_list)) if coefs[i_rule]!=0]).T
            res_=np.zeros([X.shape[0],len(rule_list)])
            res_[:,coefs!=0]=res
            return res_
    def __str__(self):
        return (map(lambda x: x.__str__(), self.rules)).__str__()




class RuleFit(BaseEstimator, TransformerMixin):
    """Rulefit class


    Parameters
    ----------
        n_feats:        The number of features
        incr_feats:     A list or array of the monotone increasing features (1 based)
        decr_feats:     A list or array of the monotone decreasing features (1 based)
        tree_size:      Number of terminal nodes in generated trees. If exp_rand_tree_size=True, 
                        this will be the mean number of terminal nodes.
        sample_fract:   fraction of randomly chosen training observations used to produce each tree. 
                        FP 2004 (Sec. 2)
        max_rules:      approximate total number of rules generated for fitting. Note that actual
                        number of rules will usually be lower than this due to duplicates.
        memory_par:     scale multiplier (shrinkage factor) applied to each new tree when 
                        sequentially induced. FP 2004 (Sec. 2)
        rfmode:         'regress' for regression or 'classify' for binary classification.
        lin_standardise: If True, the linear terms will be standardised as per Friedman Sec 3.2
                        by multiplying the winsorised variable by 0.4/stdev.
        lin_trim_quantile: If lin_standardise is True, this quantile will be used to trim linear 
                        terms before standardisation.
        exp_rand_tree_size: If True, each boosted tree will have a different maximum number of 
                        terminal nodes based on an exponential distribution about tree_size. 
                        (Friedman Sec 3.3)
        model_type:     'r': rules only; 'l': linear terms only; 'rl': both rules and linear terms
        random_state:   Integer to initialise random objects and provide repeatability.
        tree_generator: Optional: this object will be used as provided to generate the rules. 
                        This will override almost all the other properties above. 
                        Must be GradientBoostingRegressor or GradientBoostingClassifier, optional (default=None)
    Attributes
    ----------
    rule_ensemble: RuleEnsemble
        The rule ensemble

    feature_names: list of strings, optional (default=None)
        The names of the features (columns)

    """
    def __init__(self,tree_size=4,sample_fract='default',max_rules=2000,
                 memory_par=0.01,
                 tree_generator=None,n_feats=None,incr_feats=[],decr_feats=[],
                rfmode='regress',lin_trim_quantile=0.025,
                lin_standardise=True, exp_rand_tree_size=True,
                model_type='rl',random_state=None,instance_max_rules=500,
                instance_rules_per_learner=15,instance_use_exp_rand=True, instance_v=[0.001,0.01,0.05,0.25,0.5],
                instance_learner_type='one-sided',instance_learner_eta=1,
                mt_feat_mode='specified',
                auto_mt_feat_cv=5,auto_mt_feat_type='best'):
        self.tree_generator = tree_generator
        self.n_feats=n_feats
        self.incr_feats=np.asarray([] if incr_feats is None else incr_feats)
        self.decr_feats=np.asarray([] if decr_feats is None else decr_feats)
        self.mt_feats=np.asarray(list(self.incr_feats)+list(self.decr_feats))
        self.nmt_feats=np.asarray([j for j in np.arange(n_feats)+1 if j not in self.mt_feats])
        self.rfmode=rfmode
        self.lin_trim_quantile=lin_trim_quantile
        self.lin_standardise=lin_standardise
        self.friedscale=FriedScale(trim_quantile=lin_trim_quantile)
        self.exp_rand_tree_size=exp_rand_tree_size
        self.max_rules=max_rules
        self.sample_fract=sample_fract 
        self.max_rules=max_rules
        self.memory_par=memory_par
        self.tree_size=tree_size
        self.random_state=random_state
        self.model_type=model_type
        self.instance_max_rules=instance_max_rules
        self.instance_rules_per_learner=instance_rules_per_learner
        self.instance_use_exp_rand=instance_use_exp_rand
        self.instance_v=instance_v
        self.instance_learner_type=instance_learner_type
        self.mt_feat_mode=mt_feat_mode
        self.auto_mt_feat_cv=auto_mt_feat_cv
        self.glmnet_family='gaussian' if self.rfmode=='regress' else 'binomial'
        self.auto_mt_feat_type=auto_mt_feat_type
        self.instance_learner_eta=instance_learner_eta
    def load_rule_candidates(self,X,y):
        N=X.shape[0]
        ## initialise tree generator
        if self.tree_generator is None:
            n_estimators_default=int(np.ceil(self.max_rules/self.tree_size))
            self.sample_fract_=min(0.5,(100+6*np.sqrt(N))/N) if self.sample_fract=='default' else self.sample_fract    
            if   self.rfmode=='regress':
                self.tree_generator = GradientBoostingRegressor(n_estimators=n_estimators_default, max_leaf_nodes=self.tree_size, learning_rate=self.memory_par,subsample=self.sample_fract_,random_state=self.random_state,max_depth=100)
            else:
                self.tree_generator =GradientBoostingClassifier(n_estimators=n_estimators_default, max_leaf_nodes=self.tree_size, learning_rate=self.memory_par,subsample=self.sample_fract_,random_state=self.random_state,max_depth=100)

        if   self.rfmode=='regress':
            if type(self.tree_generator) not in [GradientBoostingRegressor,RandomForestRegressor]:
                raise ValueError("RuleFit only works with RandomForest and BoostingRegressor")
        else:
            if type(self.tree_generator) not in [GradientBoostingClassifier,RandomForestClassifier]:
                raise ValueError("RuleFit only works with RandomForest and BoostingClassifier")

        ## fit tree generator
        if not self.exp_rand_tree_size: # simply fit with constant tree size
            self.tree_generator.fit(X, y)
        else: # randomise tree size as per Friedman 2005 Sec 3.3
            np.random.seed(self.random_state)
            tree_sizes=np.random.exponential(scale=self.tree_size-2,size=int(np.ceil(self.max_rules*2/self.tree_size)))
            tree_sizes=np.asarray([2+np.floor(tree_sizes[i_]) for i_ in np.arange(len(tree_sizes))],dtype=int)
            i=int(len(tree_sizes)/4)
            while np.sum(tree_sizes[0:i])<self.max_rules:
                i=i+1
            tree_sizes=tree_sizes[0:i]
            self.tree_generator.set_params(warm_start=True) 
            for i_size in np.arange(len(tree_sizes)):
                size=tree_sizes[i_size]
                self.tree_generator.set_params(n_estimators=len(self.tree_generator.estimators_)+1)
                self.tree_generator.set_params(max_leaf_nodes=size)
                self.tree_generator.set_params(random_state=i_size+self.random_state) # warm_state=True seems to reset random_state, such that the trees are highly correlated, unless we manually change the random_sate here.
                self.tree_generator.get_params()['n_estimators']
                self.tree_generator.fit(np.copy(X, order='C'), np.copy(y, order='C'))
                # count leaves (a check)
            self.tree_generator.set_params(warm_start=False) 
        tree_list = self.tree_generator.estimators_
        if isinstance(self.tree_generator, RandomForestRegressor) or isinstance(self.tree_generator, RandomForestClassifier):
             tree_list = [[x] for x in self.tree_generator.estimators_]
             
        ## extract rules
        self.rule_ensemble = RuleEnsemble(tree_list = tree_list,
                                          feature_names=self.feature_names)
        return
    def load_instance_rule_candidates(self,X,y):
        N=X.shape[0]
        ## initialise instance rule generator
        self.sample_fract_=min(0.5,(100+6*np.sqrt(N))/N) if self.sample_fract=='default' else self.sample_fract    
        default_instance_v=self.instance_v #[0.001,0.05,0.25,0.5,1.0]#np.mean(self.instance_v)
        #self.instance_ensemble=MonoBoost(n_feats=self.n_feats,incr_feats=self.incr_feats,decr_feats=self.decr_feats,num_estimators=self.instance_max_rules,fit_algo='L2-one-class',eta=self.memory_par,vs=default_instance_v,verbose=False,hp_reg=None ,hp_reg_c=None ,learner_type='one-sided',v_random=True,hp_recalc_freq=self.instance_rules_per_learner,random_state=self.random_state)
        self.instance_ensemble=MonoBoostEnsemble(n_feats=self.n_feats,incr_feats=self.incr_feats,decr_feats=self.decr_feats,num_estimators=self.instance_max_rules,fit_algo='L2-one-class',eta=self.memory_par,vs=default_instance_v,verbose=False ,learner_type=self.instance_learner_type,learner_num_estimators=self.instance_rules_per_learner,learner_eta=self.instance_learner_eta, learner_v_mode='random',sample_fract =self.sample_fract_,random_state=self.random_state,standardise=True)
        self.instance_ensemble.fit(X, y)
        return 
    def get_mt_compliant_rule_ensemble(self,rule_ensemble,incr_feats,decr_feats):
        ## filter for upper and lower rules only (if needed)
        incr_feats=np.asarray(incr_feats)
        decr_feats=np.asarray(decr_feats)
        mt_feats=np.asarray(list(incr_feats)+list(decr_feats))
        filtered_rules=[]
        filtered_rules_indxs=[]
        i_rule=0
        for rule in rule_ensemble.rules:
            conditions=list(rule.conditions)
            all_incr=np.all([conditions[c].operator[0]==('>' if conditions[c].feature_index in incr_feats-1 else '<') for c in [cc for cc in np.arange(len(conditions)) if conditions[cc].feature_index in mt_feats-1]])
            all_decr=np.all([conditions[c].operator[0]==('<' if conditions[c].feature_index in incr_feats-1 else '>') for c in [cc for cc in np.arange(len(conditions)) if conditions[cc].feature_index in mt_feats-1]])
            if all_incr and all_decr: # no monotone features must be used in this rule!
                rule.rule_direction=0
                filtered_rules=filtered_rules+[rule] 
                filtered_rules_indxs=filtered_rules_indxs+[i_rule]
            if (all_incr and rule.value>0) or (all_decr and rule.value<0):
                rule.rule_direction=+1 if all_incr else -1
                filtered_rules=filtered_rules+[rule] 
                filtered_rules_indxs=filtered_rules_indxs+[i_rule]
            i_rule=i_rule+1
        #print('started with ' + str(len(self.rule_ensemble.rules)) + ' rules, now have ' + str(len(filtered_rules)))
        re=RuleEnsemble(tree_list=rule_ensemble.tree_list,feature_names=rule_ensemble.feature_names)
        re.rules=list(filtered_rules)
        return [re,filtered_rules_indxs]
    def get_lin_feats(self,X):
        if self.lin_standardise:
            self.friedscale.train(X)
            X_regn=self.friedscale.scale(X)
        else:
            X_regn=X.copy() 
        return X_regn
    def get_candidate_features(self,X):
        N=X.shape[0]
        if 'r' in self.model_type:
            X_rules = self.rule_ensemble.transform(X)
        if 'i' in self.model_type: 
            X_instance = self.instance_ensemble.transform(X)
        if 'l' in self.model_type: 
            X_regn=self.get_lin_feats(X)
           
        
        ## Compile Training data
        X_concat=np.zeros([X.shape[0],0])
        
        if 'l' in self.model_type:
            X_concat = np.concatenate((X_concat,X_regn), axis=1)
        if 'r' in self.model_type:
            if X_rules.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_rules), axis=1)
            
        if 'i' in self.model_type:
            if X_instance.shape[0] >0:
                X_concat = np.concatenate((X_concat, X_instance), axis=1)
        return X_concat
    def fit(self, X, y=None, feature_names=None):
        """Fit and estimate linear combination of rule ensemble

        """
        ## Enumerate features if feature names not provided
        if feature_names is None:
            self.feature_names = ['feature_' + str(x) for x in range(0, X.shape[1])]
        else:
            self.feature_names=feature_names
        ## load rule ensemble
        self.num_rules_peak_=0
        if 'r' in self.model_type:
            self.load_rule_candidates(X,y)
            self.num_rules_peak_=len(self.rule_ensemble.rules)
        ## Load Instance rules
        if 'i' in self.model_type: 
            self.load_instance_rule_candidates(X,y)
 
        
        # Solve Lasso
        if self.mt_feat_mode=='specified' and len(self.mt_feats)==0:
            X_concat=self.get_candidate_features(X)
            if self.rfmode=='regress':
                self.lscv = LassoCV()
            else:
                self.lscv=GLMCV(family='binomial')
        else:
            if self.mt_feat_mode=='auto':
                [self.incr_feats,self.decr_feats]=self.select_mono_feats(X,y)
            if 'r' in self.model_type: 
                [self.rule_ensemble,valid_rule_indxs]=self.get_mt_compliant_rule_ensemble(self.rule_ensemble,self.incr_feats,self.decr_feats)
            X_concat=self.get_candidate_features(X)
            [incr_feats_with_rules,decr_feats_with_rules]=self.get_candidate_feat_constraints(self.incr_feats,self.decr_feats)
            self.lscv=GLMCV(family=self.glmnet_family,incr_feats=incr_feats_with_rules,decr_feats=decr_feats_with_rules)
            
        ## fit Lasso
        self.lscv.fit(X_concat, y)
        
        return self
    def get_candidate_feat_constraints(self,incr_feats,decr_feats,override_rule_ensemble=None,override_model_type=None):
        ## Compile feature coefficient constraints
        incr_feats_with_rules=np.asarray([]) #np.hstack([self.incr_feats,self.n_feats+np.asarray([i_rule+1 for i_rule in np.arange(len(rule_dirns)) if rule_dirns[i_rule]>0])]))
        decr_feats_with_rules=np.asarray([]) #np.hstack([self.decr_feats,self.n_feats+np.asarray([i_rule+1 for i_rule in np.arange(len(rule_dirns)) if rule_dirns[i_rule]<0])]))
        n_feat_start=0
        model_type=self.model_type if override_model_type is None else override_model_type
        if 'l' in model_type:
            incr_feats_with_rules=incr_feats
            decr_feats_with_rules=decr_feats
            n_feat_start=self.n_feats
        if 'r' in model_type:
            re=self.rule_ensemble if override_rule_ensemble is None else override_rule_ensemble
            rule_dirns=np.asarray([r.rule_direction for r in re.rules])
            incr_feats_with_rules=np.hstack([incr_feats_with_rules,n_feat_start+np.asarray([i_rule+1 for i_rule in np.arange(len(rule_dirns)) if rule_dirns[i_rule]>0])])
            decr_feats_with_rules=np.hstack([decr_feats_with_rules,n_feat_start+np.asarray([i_rule+1 for i_rule in np.arange(len(rule_dirns)) if rule_dirns[i_rule]<0])])
            n_feat_start=n_feat_start+len(re.rules)
        if 'i' in model_type:
            instance_dirns=np.asarray([r.dirn for r in self.instance_ensemble.get_all_learners()])
            incr_feats_with_rules=np.hstack([incr_feats_with_rules,n_feat_start+np.asarray([i_rule+1 for i_rule in np.arange(len(instance_dirns)) if instance_dirns[i_rule]>0])])
            decr_feats_with_rules=np.hstack([decr_feats_with_rules,n_feat_start+np.asarray([i_rule+1 for i_rule in np.arange(len(instance_dirns)) if instance_dirns[i_rule]<0])])
            n_feat_start=n_feat_start+len(self.instance_ensemble.estimators)
        
        return [incr_feats_with_rules,decr_feats_with_rules]
    def predict (self, X):
        """Predict outcome for X

        """

        X_concat = self.get_candidate_features(X)#
        return self.lscv.predict(X_concat)

    def transform(self, X=None, y=None):
        """Transform dataset.

        Parameters
        ----------
        X : array-like matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency.

        Returns
        -------
        X_transformed: matrix, shape=(n_samples, n_out)
            Transformed data set
        """
        return self.rule_ensemble.transform(X)

    def get_rules(self, exclude_zero_coef=True):
        """Return the estimated rules

        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.

        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """

        n_features= self.n_feats#len(self.lscv.coef_) - len(self.rule_ensemble.rules)
        
        output_rules = []
        n_up_to=0
        if 'l' in self.model_type:
            ## Add coefficients for linear effects
            for i in range(0, n_features):
                if self.lin_standardise:
                    coef=self.lscv.coef_[i ]*self.friedscale.scale_multipliers[i]
                else:
                    coef=self.lscv.coef_[i ]
                output_rules += [(self.feature_names[i], 'linear',coef, 1,0)]
            n_up_to=n_up_to+n_features
        ## Add rules
        if 'r' in self.model_type:
            rule_ensemble = list(self.rule_ensemble.rules)
            for i in range(0, len(self.rule_ensemble.rules)):
                rule = rule_ensemble[i]
                coef=self.lscv.coef_[i + n_up_to]
                output_rules += [(rule.__str__(), 'rule', coef,  rule.support,rule.rule_direction)]
            n_up_to=n_up_to+len(rule_ensemble)
        ## Add rules
        if 'i' in self.model_type:
            instance_ensemble = list(self.instance_ensemble.get_all_learners())
            for i in range(0, len(instance_ensemble)):
                rule = instance_ensemble[i]
                coef=self.lscv.coef_[i + n_up_to]
                output_rules += [(rule.__str__(), 'instance', coef,  99,rule.dirn)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support","dirn"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules

    def select_mono_feats(self,X_train,y_train):
        loss_fn=rmse if self.rfmode=='regress' else binom_dev
        X_rules_all = self.rule_ensemble.transform(X_train) # transform all rules once here, then filter appropriate columns below
        X_regn=self.get_lin_feats(X_train)
        last_overall_best=[1e9+1,[],[]]
        overall_best=[1e9,[],[]]
        while overall_best[0]<last_overall_best[0]:
            test_best=overall_best.copy()
            feats=np.arange(X_train.shape[1]+1) if len(overall_best[1])==0 and len(overall_best[2])==0 else np.arange(X_train.shape[1])+1
            for feat in feats:
                if feat not in overall_best[1] +overall_best[2]: 
                    feat_losses=[0,0]
                    for dirn in [0] if feat==0 else [-1,+1]:
                        if feat==0: # include the no MT feature case
                            incr_feats_test=[]
                            decr_feats_test=[]
                        else:
                            incr_feats_test=overall_best[1]+([feat] if dirn==1 else [])
                            decr_feats_test=overall_best[2]+([feat] if dirn==-1 else [])
                        [rule_ensemble,valid_rule_indxs]=self.get_mt_compliant_rule_ensemble(self.rule_ensemble,incr_feats_test,decr_feats_test)
                        X_rules = X_rules_all[:,valid_rule_indxs]
                        X_concat = np.concatenate((X_regn, X_rules), axis=1)
                        [incr_feats_with_rules,decr_feats_with_rules]=self.get_candidate_feat_constraints(incr_feats_test,decr_feats_test,override_rule_ensemble=rule_ensemble,override_model_type='rl')
                        #print(str(incr_feats_test) + ' ' + str(decr_feats_test) + ' ' + str(len(valid_rule_indxs)) +' ' + str(len(incr_feats_with_rules)+len(decr_feats_with_rules)))
                        # estimate performance in CV loop
                        kf = StratifiedKFold(n_folds=self.auto_mt_feat_cv,y=y_train,shuffle=False)
                        pred_y_proba=np.zeros(len(y_train))
                        for train_index, test_index in kf: #kf.split(X_train, y_train):
                            X_train_cv, X_test_cv = X_concat[train_index,:], X_concat[test_index,:]
                            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
                            lscv_cv=GLMCV(family=self.glmnet_family,incr_feats=incr_feats_with_rules,decr_feats=decr_feats_with_rules)
                            lscv_cv.fit(X_train_cv, y_train_cv)
                            pred_y_proba[test_index]=lscv_cv.predict_proba(X_test_cv)
                        loss=loss_fn(y_train,pred_y_proba)
                        feat_losses[0 if dirn==-1 else 1]=loss
                        print(str(incr_feats_test) + ' ' + str(decr_feats_test) + ': ' + str(loss))
                        if dirn==0: # baseline, no mt feats
                            overall_best[0]=loss
                        if self.auto_mt_feat_type=='best' or dirn==0:
                            if loss<test_best[0]:
                                test_best=[loss,incr_feats_test,decr_feats_test]
                        elif self.auto_mt_feat_type=='best_diff':
                            if dirn==1:
                                best=np.argmin(feat_losses)
                                worst=np.argmax(feat_losses)
                                if feat_losses[best]<test_best[0] and feat_losses[worst]>=overall_best[0] :
                                    test_best=[feat_losses[best],overall_best[1]+([feat] if best==1 else []),overall_best[2]+([feat] if best==0 else [])]
                            
            # if best is an improvement, add to appropriate direction, else stop
            last_overall_best=overall_best.copy()
            if test_best[0]<overall_best[0]:
                overall_best=test_best.copy()
                print('    UPDATE: ' + str(test_best))
                if len(overall_best[1])==0 and len(overall_best[2])==0: # this means that no MT feats was the best option after considering all!
                    break
                                
        # return optimal features
        incr_feats=overall_best[1]
        decr_feats=overall_best[2]
        return [incr_feats,decr_feats]
        
def binom_dev(y_true,y_pred):
    return np.sum(np.log(1+np.exp(-2*y_true*y_pred)))
    
def rmse(y_true,y_pred):
    return -np.sqrt(np.sum((y_true-y_pred)**2))
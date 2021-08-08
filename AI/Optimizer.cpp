#pragma once
#include <unordered_map>
#include "OperationGraph.cpp"
#include "device_functions.cu"

//TODO: Clip gradients
//TODO: Weight decay, L1, L2
//TODO: https://miro.medium.com/max/407/1*N8QAtWmiFPL15dGfrAKPRw.png
//TODO: https://medium.com/geekculture/a-2021-guide-to-improving-cnns-recent-optimizers-a340456f6b2d

//============================================================
//==================|Declare new classes|=====================
//============================================================
enum OPTIMIZER_TYPE : uint32_t { OPTIMIZER_NONE = 0, OPTIMIZER_DBUG = 1, OPTIMIZER_SGD = 2, OPTIMIZER_SGD_MOMENT = 3, OPTIMIZER_ADAM = 4 };
class Optimizer;
class Optimizer_None;
class Optimizer_SGD;
class Optimizer_Debug;
class Optimizer_Adam;


#ifdef TODO
//===================================================
//==================|Line search|====================
//===================================================

enum LINE_SEARCH : uint16_t { NONE = 0, BACKTRACKING_ARMIJO = 1 };
class BacktrackingLineSearch {
	/*
		Performs a onedimensional backtracking linesearch (adds it to graph)
	*/
protected:
	TYPE data_type;

	struct BufAndStep {
		TensorDesc* buf;
		TensorDesc* step;
	};
	std::vector<BufAndStep> bufsAndSteps;

	double startVal;
	double decayFac;

	OperationGraph* forwardProp;
	TensorDesc* lossVal;
	ConstantDesc* stepSize;

public:
	BacktrackingLineSearch(TYPE data_type) :
		data_type(data_type),
		bufsAndSteps(),
		startVal(0.),
		decayFac(0.),
		forwardProp(nullptr),
		lossVal(nullptr),
		stepSize(nullptr)
	{}

	void registerVars(TensorDesc* vars, TensorDesc* steps) {
		bufsAndSteps.push_back({ vars, steps });
	}

	template<typename T>
	void setConstants(T startVal_, T decayFac_) {
		startVal->setValue(startVal_);
		decayFac->setValue(decayFac_);
	}

	void setForwardOperation(OperationGraph* forwd, TensorDesc* lossBuf) {
		forwardProp = forwd;
		lossVal = lossBuf;
	}

	OperationGraph* getUpdateGraph() {
		/*
		TensorDesc* stepSize = (new TensorDesc())
			->setTypeId(data_type)
			->setSize({ 1u, 1u, 1u, 1u, 1u })
			->setStrides({ 1u, 1u, 1u, 1u, 1u })
			->setAlignment(16u)
			->setIsNeeded(false)
			->setUseIndirection(false);
		*/

		OperationGraph* graph = new OperationGraph();
		for (BufAndStep& e : bufsAndSteps) {
			graph->addNode((new Operation_Copy())
				->setInTensor(e.step)
				->setOutTensor(e.buf)
				->setBlendConstants({
						stepSize,
						(new ConstantDesc(data_type))->setUseIndirection(false)->setValue(1.0)
					})
			);
		}

		return graph;
	}
};
#endif

//===================================================
//==================|Optimizers|=====================
//===================================================
//TODO: Let user register hyperparams as constant so ConstantDesc can be registered as constant and inlined in OperationGraph
class Optimizer {
public:
	TYPE data_type;        //The type of the variables the optimizer optimizes
	TYPE computation_type; //The type used for computations

protected:
	std::unordered_map<std::string, double>                                       hyperparams;
	std::unordered_map<std::string, ConstantDesc*>                                constants; 
	std::unordered_map<TensorDesc*, std::unordered_map<std::string, TensorDesc*>> buffers;   
	
public:
	//Constructors
	Optimizer() {
		throw new std::runtime_error("[ERROR] Trying to create a base class optimizer");
	}

	//Methods
	/*
		Adds nodes to operation graph that update memory using deltas

		@param mem           : The memory to update
		@param deltas        : The deltas(gradients) to use to update the memory. Needs to have the same shape as mem
		@param scaleInvariant: True, if it scaling mem with a factor does not change forward prop. Often the case due to batch norm
	*/
	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas, bool scaleInvariant = false) {
		throw new std::runtime_error("[ERROR] Trying to call \"applyDeltaToMemory\" on base class optimizer");
	}

	/*
		Initializes the internal optimization buffers. This has to be called after the operation graph was compiled as before that no memory was allocated
	*/
	virtual void initMem() {
		throw new std::runtime_error("[ERROR] Trying to call \"initMem\" on base class optimizer!");
	}

	/*
		Recalculates the values of ConstantDesc's in constants based on values in hyperparams.
		Hyperparams can be manipulated using "getHyperparamPointer"
	*/
	virtual void recalculateGpuConstants() {
		throw new std::runtime_error("[ERROR] Trying to call \"recalculateGpuConstants\" on base class optimizer!");
	}

	/*
		Exposes hyperparams to user by returning pointer to them so user can read/write to them.
	*/
	std::unordered_map<std::string, double>* getHyperparamPointer() const {
		return &hyperparams;
	}

	//Serialization
	/*
		Return the derived class type of the object
	*/
	virtual OPTIMIZER_TYPE getType() const {
		throw new std::runtime_error("[ERROR] Trying to call \"getType\" on base class optimizer");
	}
	/*
		Returns a pointer to a newly created default initialized object of the specified derived class

		@param type: The derived class of the returned object
	*/
	inline static Optimizer* getOptimizerOfType(OPTIMIZER_TYPE type) {
		switch (type) {
		case OPTIMIZER_TYPE::OPTIMIZER_NONE:
			return new Optimizer_None();
			break;
		case OPTIMIZER_TYPE::OPTIMITER_DBUG:
			return new Optimizer_Debug();
			break;
		case OPTIMIZER_TYPE::OPTIMIZER_SGD:
			return new Optimizer_SGD();
			break;
		case OPTIMIZER_TYPE::OPTIMIZER_ADAM:
			return new Optimizer_Adam();
			break;
		default:
			throw new std::runtime_error("[ERROR] Trying to create an optimizer of unsupported type");
		}
	}
	/*
		Serializes the type of this optimizer to the specified file. The information can be deserialized using the "deserialize" method

		@param file: FILE* to an already opened file where the information is written to
	*/
	inline void serialize(FILE* file) const {
		OPTIMIZER_TYPE type = getType();
		fwrite(&type, sizeof(type), 1, file);
	}
	/*
		Deserializes the type of this optimizer from the specified file. Return an object of this type. The file has to have been created using the "serialize" method

		@param file: FILE* to an already opened file where the information is read to
	*/
	inline static Optimizer* deserialize(FILE* file) {
		OPTIMIZER_TYPE type;
		fread(&type, sizeof(type), 1, file);
		return getOptimizerOfType(type);
	}
};


enum MOMENTUM       : uint16_t { NONE = 0, VANILLA = 1, NESTROV = 2 };
enum ADAPTIVE       : uint16_t { NONE = 0, SQRT = 1, CBRT = 2 };
enum DEBIAS         : uint16_t { NONE = 0, APPROX = 1, EXACT = 2 };
enum RECTIFY        : uint16_t { NONE = 0, APPROX = 1, EXACT = 2 };
enum CLIPPING       : uint16_t { NONE = 0, GRADIENT = 1, STEP = 2 };
enum CENTRALIZATION : uint16_t { NONE = 0, GRADIENT = 1, STEP = 2 };
enum DEBUG_INFO     : uint16_t { NONE = 0, GRADIENT = 1, STEP = 2, MOMENTUM_BUF = 3, ADAPTIVE_BUF = 4 };
enum WEIGHT_DECAY   : uint16_t { NONE = 0, L1 = 0b01, L2 = 0b10, GRAD_PLAIN = 1 << 2, GRAD_VANILLA_LR = 2 << 2, STEP_PLAIN = 3 << 2, STEP_VANILLA_LR = 4 << 2, STEP_EFFECTIVE_LR = 5 << 2, STEP_EFFECTIVE_LR_MEAN = 6 << 2};
WEIGHT_DECAY operator^(WEIGHT_DECAY a, WEIGHT_DECAY b) { return (WEIGHT_DECAY)((uint16_t)a ^ (uint16_t)b); }
WEIGHT_DECAY operator&(WEIGHT_DECAY a, WEIGHT_DECAY b) { return (WEIGHT_DECAY)((uint16_t)a & (uint16_t)b); }
WEIGHT_DECAY operator>>(WEIGHT_DECAY a, uint32_t s) { return (WEIGHT_DECAY)((uint16_t)a >> s); }

//TODO: Madgrad, Adam2k, Adamax, Heavyball, Diffgrad, Adamod, Adabelieve, LARS, LAMB
//TODO: Line search
class FirtOrder_Optimizer : public Optimizer {
	/*
		Optimizes a set of variables with only their first order gradient information
	*/

	/*
			g += "weightDecay" * w
			g = centralize(clip(g))
			m = "momentumDecay" * m + "momentumNew" * g
			n = "adaptiveDecay" * n + "adaptiveNew" * (g*g)
			n_deb = bound("debiasAdaptive" / f(n + eps))
			fac = ("debiasMomentum" * m + "nestrovFactor" * g) * n_deb [ + "weightDecay" * w] [ + "weightDecay" * mean(n_deb) * w]
			step = centralize(project(clip("stepSize" * fac [ + "weightDecay" * w])))
			"stepSize" = "oldStepSizeDecay" * "stepSize" + "oldStepSizeNew" * step
			save debug
			w -= step
			if(lookaheadSyncBool) w = w_old + "lookaheadStepSize" * (w - w_old); w_old = w

			Optimizations:
			 - fused nestrov and debias for momentum
			 - debiasAdaptive needed if (debias or nestrov or rectify) and (no momentum or bounding).
			 - effective lr mean inlines debiasing if needed
			 - stepSize needed if !deltaLR && !nestrov && !debias
		*/

private:
	MOMENTUM arg_momentum        ;
	ADAPTIVE arg_adaptive        ;
	WEIGHT_DECAY arg_decay       ;
	CLIPPING arg_clipping        ;
	bool arg_boundAdaptive       ;
	DEBIAS arg_deBias            ;
	CENTRALIZATION arg_centralize;
	LINE_SEARCH arg_lineSearch   ;
	uint32_t arg_lookahead       ;
	bool arg_deltaLr             ;
	bool arg_projection          ;
	bool arg_rectify             ;
	DEBUG_INFO arg_debug         ;

public:
	FirstOrder_Optimizer(
		TYPE data_type,
		TYPE computation_type     = data_type,
		MOMENTUM momentum         = MOMENTUM::NESTROV,
		ADAPTIVE adaptive         = ADAPTIVE::NONE,
		WEIGHT_DECAY decay        = WEIGHT_DECAY::L2 ^ WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN,
		CLIPPING clipping         = CLIPPING::GRADIENT, 
		bool boundAdaptive        = false,
		DEBIAS deBias             = DEBIAS::NONE, 
		CENTRALIZATION centralize = CENTRALIZATION::GRADIENT,
		LINE_SEARCH lineSearch    = LINE_SEARCH::NONE,
		uint32_t lookahead        = 0u,
		bool deltaLr              = false,
		bool projection           = true,
		bool rectify              = false,
		DEBUG_INFO debug          = DEBUG_INFO::NONE 
	) :
		arg_momentum     (momentum     ), 
		arg_adaptive     (adaptive     ),
		arg_decay	     (decay	       ),
		arg_clipping     (clipping     ),
		arg_boundAdaptive(boundAdaptive),
		arg_deBias	     (deBias	   ),
		arg_centralize   (centralize   ),
		arg_lineSearch   (lineSearch   ),
		arg_lookahead    (lookahead    ),
		arg_deltaLr	     (deltaLr      ),
		arg_projection   (projection   ),
		arg_rectify      (rectify      ),
		arg_debug        (debug        )
	{
		//0.: Check consistency
		if (debug == DEBUG_INFO::MOMENTUM_BUF && momentum == MOMENTUM::NONE)
			throw new std::runtime_error("[ERROR] Optimizer can't save momentum buffer if no momentum is used!");
		if (debug == DEBUG_INFO::ADAPTIVE_BUF && adaptive == ADAPTIVE::NONE)
			throw new std::runtime_error("[ERROR] Optimizer can't save adaptive buffer if no adaptivity is used!");
		if (decay != WEIGHT_DECAY::NONE && !(decay & (WEIGHT_DECAY::L1 | WEIGHT_DECAY::L2)))
			throw new std::runtime_error("[ERROR] If optimizer uses weight decay, it must be either L1 or L2!");
		if (((decay & ~0b11) == WEIGHT_DECAY::STEP_EFFECTIVE_LR || (decay & ~0b11) == WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN) && adaptive == ADAPTIVE::NONE )
			throw new std::runtime_error("[ERROR] If optimizer uses weight decay based on effective lr, it also needs to use adaptive lr!");
		if (boundAdaptive && !adaptive)
			throw new std::runtime_error("[ERROR] If optimizer wants to bound the adaptive lr, it has to use adaptive lr!");
		if (rectify && !adaptive)
			throw new std::runtime_error("[ERROR] If optimizer wants to rectify the adaptive lr, it has to use adaptive lr!");
		if (debias != DEBIAS::NONE && momentum == MOMENTUM::NONE && adaptive == ADAPTIVE::NONE)
			throw new std::runtime_error("[ERROR] If optimizer debias, it has to use momentum or adapitve!");

		//1.: Set base class member variables
		this->data_type        = data_type;
		this->computation_type = computation_type;
		this->constants        = std::unordered_map<std::string, ConstantDesc*>;
		this->buffers          = std::unordered_map<TensorDesc*, std::unordered_map<std::string, TensorDesc*>>();
		this->hyperparams      = std::unordered_map<std::string, double>();

		constexpr double Nan = std::nan("0");
		if (debias != DEBIAS::NONE) this->hyperparams.emplace("timeStep"     , Nan);
		if (deltaLr) {
			this->hyperparams.emplace("deltaNew"  , Nan);
			this->hyperparams.emplace("deltaDecay", Nan);
		}
		else this->hyperparams.emplace("stepSize"     , Nan);
		if (momentum != MOMENTUM::NONE){
			this->hyperparams.emplace("momentumNew"  , Nan);
			this->hyperparams.emplace("momentumDecay", Nan);
		}
		if (momentum == MOMENTUM::NESTROV) this->hyperparams.emplace("momentumDecayNext", Nan);
		if (adaptive != ADAPTIVE::NONE) {
			this->hyperpams.emplace("adaptiveNew"  , Nan);
			this->hyperpams.emplace("adaptiveDecay", Nan);
			this->hyperpams.emplace("epsilon"      , Nan);
		}
		if (clipping != CLIPPING::NONE) {
			this->hyperparams.emplace("clipMin", Nan);
			this->hyperparams.emplace("clipMax", Nan);
		}
		if (boundAdaptive) {
			this->hyperparams.emplace("boundMin", Nan);
			this->hyperparams.emplace("boundMax", Nan);
		}
		if (decay != WEIGHT_DECAY::NONE) this->hyperparams.emplace("weightDecay", Nan);


		//2.: Set helper variables
		bool momentumNeedFactor = ((arg_momentum == MOMENTUM::NESTROV) || (arg_momentum == MOMENTUM::VANILLA && arg_debias));
		bool momentumHasFactor = momentumNeedFactor;

		bool adaptiveNeedFactor = (arg_adaptive != ADAPTIVE::NONE) && (arg_debias != DEBIAS::NONE || arg_rectify != RECTIFY::NONE || arg_momentum == MOMENTUM::NESTROV);
		bool adaptiveNeedInnerFactor = adaptiveNeedFactor && arg_boundAdaptive;
		bool adaptiveHasInnerFactor = adaptiveNeedInnerFactor;
		bool adaptiveHasOuterFactor = !adaptiveNeedInnerFactor && adaptiveNeedFactor && !momentumHasFactor;
		bool adaptiveHasFactor = adaptiveHasInnerFactor || adaptiveHasOuterFactor;
		bool inlineAdaptive = adaptiveNeedFactor && !adaptiveHasFactor;

		bool inlineStepSize = !arg_deltaLr && (momentumHasFactor || adaptiveHasOuterFactor);

		//3.: Set up constants		
		if (deltaLr) {
			this->constants.emplace("oldStepSizeDecay", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("oldStepSizeNew"  , (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}
		else if (!inlineStepSize) {
			this->constants.emplace("stepSize", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}

		if (momentum != MOMENTUM::NONE) {
			this->constants.emplace("momentumDecay", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("momentumNew"  , (new ConstantDesc(this->data_type))->setUseIndirection(true));
			
			if (momentumHasFactor)
				this->constants.emplace("momentumDebias", (new ConstantDesc(this->data_type))->setUseIndirection(true));

			if (momentum == MOMENTUM::NESTROV)
				this->constants.emplace("nestrovFactor", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}

		if (adaptive != ADAPTIVE::NONE) {
			this->constants.emplace("adaptiveDecay", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("adaptiveNew", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("epsilon", (new ConstantDesc(this->data_type))->setUseIndirection(true));

			if (adaptiveHasFactor)
				this->constants->emplace("adaptiveDebias", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}

		if (boundAdaptive) {
			this->constants.emplace("adaptiveBoundMin", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("adaptiveBoundMax", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}

		if (clipping) {
			this->constants.emplace("clippingMin", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("clippingMax", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}

		if (weight_decay != WEIGHT_DECAY::NONE) {
			this->constants.emplace("weightDecay", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}

		if (rectify) {
			this->constants.emplace("rectificationBool", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			//Factor is always inlined
		}

		if (lookahead > 0u) {
			this->constants.emplace("lookaheadSyncBool", (new ConstantDesc(this->data_type))->setUseIndirection(true));
			this->constants.emplace("lookaheadStepSize", (new ConstantDesc(this->data_type))->setUseIndirection(true));
		}
	}

	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas, bool scaleInvariant = false) override {
		//0.: Check input
		if (this->data_type != mem->getTypeId())    throw new std::runtime_error("[ERROR] Optimizer requires memory to be of the previously specified data type");
		if (this->data_type != deltas->getTypeId()) throw new std::runtime_error("[ERROR] Optimizer requires deltas to be of the previously specified data type");

		//1.: Set up buffers
		if (this->buffers.contains(mem)) throw new std::runtime_error("[ERROR] This optimizer was already called on these variables before!");
		std::unordered_map<std::string, TensorDesc*> optBufs;
		if (momentum != MOMENTUM::NONE)	optBufs.emplace("momentum", (new TensorDesc(mem))->setIsNeeded(true));
		if (adaptive != ADAPTIVE::NONE) optBufs.emplace("adaptive", (new TensorDesc(mem))->setIsNeeded(true));
		if (lookahead)                  optBufs.emplace("lookahead", (new TensorDesc(mem))->setIsNeeded(true));
		if (deltaLr)                    optBufs.emplace("deltaLr", (new TensorDesc(mem))->setIsNeeded(true));
		if (debug)                      optBufs.emplace("debug", (new TensorDesc(mem))->setIsNeeded(true));

		//1.5.: Helper variables
		ConstantDesc* ZERO = (new ConstantDesc(this->data_type))->setUseIndirection(false)->setValue(0.);
		ConstantDesc* ONE  = (new ConstantDesc(this->data_type))->setUseIndirection(false)->setValue(1.);

		bool momentumNeedFactor = ((arg_momentum == MOMENTUM::NESTROV) || (arg_momentum == MOMENTUM::VANILLA && arg_debias));
		bool momentumHasFactor = momentumNeedFactor;

		bool adaptiveNeedFactor = (arg_adaptive != ADAPTIVE::NONE) && (arg_debias != DEBIAS::NONE || arg_rectify != RECTIFY::NONE || arg_momentum == MOMENTUM::NESTROV);
		bool adaptiveNeedInnerFactor = adaptiveNeedFactor && arg_boundAdaptive;
		bool adaptiveHasInnerFactor = adaptiveNeedInnerFactor;
		bool adaptiveHasOuterFactor = !adaptiveNeedInnerFactor && adaptiveNeedFactor && !momentumHasFactor;
		bool adaptiveHasFactor = adaptiveHasInnerFactor || adaptiveHasOuterFactor;
		bool inlineAdaptive = adaptiveNeedFactor && !adaptiveHasFactor;

		bool inlineStepSize = !arg_deltaLr && (momentumHasFactor || adaptiveHasOuterFactor);

		//2.: Compute lr
		TensorDesc* oldStepsRoot = new TensorDesc(optBufs["deltaLr"]);
		if (arg_deltaLr) {
			graph->addNode((new Operation_Pointwise())
				->setPointwiseType(POINTWISE_TYPE::SQRT)
				->setInTensor(optBufs["deltaLr"])
				->setOutTensor(oldStepsRoot)
				->setBlendConstants({ ONE, ZERO })
			);
		}

		//2.: Regularization pre update
		//2.1: Clipping
		if (arg_clipping == CLIPPING::GRADIENT) {
			graph->addNode((new Operation_Pointwise())
				->setInTensor(deltas)
				->setOutTensor(deltas)
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_CLIP)
				->setArgument({ this->constants["clippingMin"], this->constants["clippingMax"] })
				->setBlendConstants({ ONE, ZERO })
			);
		}

		//2.2.: Gradient weight decay
		if ((arg_decay & ~0b11) == WEIGHT_DECAY::GRAD_PLAIN ||
			((arg_decay & ~0b11) == WEIGHT_DECAY::GRAD_VANILLA_LR && !arg_deltaLr)) {
			if (decay & WEIGHT_DECAY::L1) { //L1
				graph->addNode((new Operation_Pointwise())
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setInTensor(mem)
					->setOutTensor(delta)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
			else { //L2
				graph->addNode((new Operation_Copy())
					->setInTensor(mem)
					->setOutTensor(delta)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
		}
		if ((arg_decay & ~0b11) == WEIGHT_DECAY::GRAD_VANILLA_LR && arg_deltaLr) {
			if (decay & WEIGHT_DECAY::L1) { //L1
				TensorDesc* memSgn = new TensorDesc(mem);
				graph->addNode((new Operation_Pointwise())
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setInTensor(mem)
					->setOutTensor(memSgn)
					->setBlendConstants({ ONE, ZERO })
				);
				graph->addNode((new Operation_Binary())
					->setBinaryType(BINARY_TYPE::BINARY_MUL)
					->setInTensors(memSgn, oldStepsRoot)
					->setOutTensor(delta)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
			else { //L2
				graph->addNode((new Operation_Binary())
					->setBinaryType(BINARY_TYPE::BINARY_MUL)
					->setInTensors(mem, oldStepsRoot)
					->setOutTensor(delta)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
		}

		//2.3.: Centralization
		if (arg_centralize == CENTRALIZATION::GRADIENT) {
			TensorDesc* gradient_mean = (new TensorDesc())
				->setTypeId(this->data_type)
				->setSize({ deltas->getSize()[0], deltas->getSize()[1], 1u, 1u, 1u })
				->setStrides({ deltas->getSize()[0] * deltas->getSize()[1], deltas->getSize()[1], 1u, 1u, 1u })
				->setAlignment(deltas->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			
			graph->addNode((new Operation_Reduce())     //Compute mean
				->setReduceType(REDUCE_TYPE::MEAN)
				->setInTensor(deltas)
				->setOutTensor(gradient_mean)
				->setReduceDimensions({ 2u,3u,4u }) //All spacial dimension
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Fill()) //Fill gradient
				->setConstant(gradient_mean)
				->setOutTensor(deltas)
				->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0),
					ONE
				})
			);
		}

		//3.: Update gradient buffers
		//3.1.: Momentum buffer
		if (arg_momentum != MOMENTUM::NONE) {
			graph->addNode((new Operation_Copy())
				->setInTensor(deltas)
				->setOutTensor(optBufs["momentum"])
				->setBlendConstants({ this->constants["momentumNew"], this->constants["momentumDecay"] })
			);
		}

		//3.2.: Adaptive buffer
		if (arg_adaptive != ADAPTIVE::NONE) {
			graph->addNode((new Operation_Pointwise())
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_SQUARE)
				->setInTensor(deltas)
				->setOutTensor(optBufs["adaptive"])
				->setBlendConstants({ this->constants["adaptiveNew"], this->constants["adaptiveDecay"] })
			);
		}

		//4.: Build step buffer
		TensorDesc* step = new TensorDesc(mem); //This holds the step every weights is moved in (actually this is the value that will be subtracted from every weight)

		//4.1: Set gradient/momentum.
		switch (arg_momentum) {
		case MOMENTUM::NONE:
			graph->addNode((new Operation_Copy())
				->setInTensors(deltas)
				->setOutTensor(step)
				->setBlendConstants({ ONE, ZERO })
			);
			break;

		case MOMENTUM::NESTROV:
			graph->addNode((new Operation_Copy())
				->setInTensor(optBufs["momentum"])
				->setOutTensor(step)
				->setBlendConstants({ this->constants["momentumDebias"], ZERO})
			);
			graph->addNode((new Operation_Copy())
				->setInTensor(deltas)
				->setOutTensor(step)
				->setBlendConstants({ this->constants["nestrovFactor"], ONE})
			);
			break;

		case MOMENTUM::VANILLA:
			graph->addNode((new Operation_Copy())
				->setInTensor(optBufs["momentum"])
				->setOutTensor(step)
				->setBlendConstants({
					(arg_debias) ? this->constants["momentumDebias"] : ONE,
					ZERO
				})
			);
			break;
		}

		//4.2.: Effective lr decay
		if ((arg_decay & ~0b11) == WEIGHT_DECAY::STEP_EFFECTIVE_LR) {
			if (decay & WEIGHT_DECAY::L1) { //L1
				graph->addNode((new Operation_Pointwise())
					->setInTensor(mem)
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setOutTensor(step)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
			else { //L2
				graph->addNode((new Operation_Copy())
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
		}
		
		//4.3.: Compute adaptive lr with rectification
		TensorDesc* adaptiveMultiplier = new TensorDesc(optBufs["adaptive"]);
		std::vector<Operation*> adaptiveOps;
		switch (arg_adaptive) {
		case ADAPTIVE::NONE:
			break;
		case ADAPTIVE::SQRT:
			adaptiveOps.push_back((new Operation_Copy())  //Debias if needed, else copy
				->setInTensor(optBufs["adaptive"])
				->setOutTensor(adaptiveMultiplier)
				->setBlendConstants({
					adaptiveHasFactor ? this->constants["adaptiveDebias"] : ONE,
					ZERO
				})
			);
			adaptiveOps.push_back((new Operation_Pointwise()) //Add epsilon
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_ADD)
				->setInTensor(adaptiveMultiplier)
				->setOutTensor(adaptiveMultiplier)
				->setArgument(this->constants["epsilon"])
				->setBlendConstants({ ONE, ZERO })
			);
			adaptiveOps.push_back((new Operation_Pointwise()) //1/sqrt(x)
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_ROOT_REC)
				->setInTensor(adaptiveMultiplier)
				->setOutTensor(adaptiveMultiplier)
				->setBlendConstants({ ONE, ZERO })
			);

			if (arg_boundAdaptive) {
				adaptiveOps.push_back((new Operation_Pointwise()) //Bound adaptive
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_CLIP)
					->setInTensor(adaptiveMultiplier)
					->setOutTensor(adaptiveMultiplier)
					->setArguments({ this->constants["adaptiveBoundMin"], this->constants["adaptiveBoundMax"] })
					->setBlendConstants({ ONE, ZERO })
				);
			}

			adaptiveOps.push_back(new Operation_Binary())
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(adaptiveMultiplier, step)
				->setOutTensor(step)
				->setBlendConstants({ ONE, ZERO })
				);
				break;
		case ADAPTIVE::CBRT:
			adaptiveOps.push_back((new Operation_Copy())  //Debias if needed, else copy
				->setInTensor(optBufs["adaptive"])
				->setOutTensor(adaptiveMultiplier)
				->setBlendConstants({
					adaptiveHasFactor ? this->constants["adaptiveDebias"] : ONE,
					ZERO
				})
			);
			adaptiveOps.push_back((new Operation_Pointwise()) //Add epsilon
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_ADD)
				->setInTensor(adaptiveMultiplier)
				->setOutTensor(adaptiveMultiplier)
				->setArgument(this->constants["epsilon"])
				->setBlendConstants({ ONE, ZERO })
			);
			adaptiveOps.push_back((new Operation_Pointwise()) //1/cbrt(x)
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_POW)
				->setInTensor(adaptiveMultiplier)
				->setOutTensor(adaptiveMultiplier)
				->setArgument({ new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0 / 3.0) })
				->setBlendConstants({ ONE, ZERO })
			);

			if (arg_boundAdaptive) {
				adaptiveOps.push_back((new Operation_Pointwise()) //Bound adaptive
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_CLIP)
					->setInTensor(adaptiveMultiplier)
					->setOutTensor(adaptiveMultiplier)
					->setArguments({ this->constants["adaptiveBoundMin"], this->constants["adaptiveBoundMax"] })
					->setBlendConstants({ ONE, ZERO })
				);
			}

			adaptiveOps.push_back(new Operation_Binary())
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(adaptiveMultiplier, step)
				->setOutTensor(step)
				->setBlendConstants({ ONE, ZERO })
				);
				break;
		}
		
		if (arg_rectification) {
			Operation_If* useRectification = (new Operation_If())
				->setConditionConstant(this->constants["rectificationBool"]);

			for (Operation*& op : adaptiveOps)
				useRectification->addOperationTrue(op);
			
			useRectification->addOperationFalse((new Operation_Fill())
				->setConstant(ONE)
				->setInTensor(adaptiveMultiplier);
			);

			graph->addOperation(useRectification);
		}
		else {
			for (Operation*& op : adaptiveOps)
				graph->addNode(op);
		}

		//4.4.: Effective lr mean & vanilla lr decay
		if ((arg_decay & ~0b11) == WEIGHT_DECAY::STEP_VANILLA_LR) {
			if (decay & WEIGHT_DECAY::L1) { //L1
				graph->addNode((new Operation_Pointwise())
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
			else { //L2
				graph->addNode((new Operation_Copy())
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
		}
		else if ((arg_decay & ~0b11) == WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN) {
			TensorDesc* adaptiveMean = (new TensorDesc())
				->setTypeId(this->data_type)
				->setSize({ adaptiveMultiplier->getSize()[0], adaptiveMultiplier->getSize()[1], 1u, 1u, 1u })
				->setStrides({ adaptiveMultiplier->getSize()[0] * adaptiveMultiplier->getSize()[1], adaptiveMultiplier->getSize()[1], 1u, 1u, 1u })
				->setAlignment(adaptiveMultiplier->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			TensorDesc* adaptiveMeanFilled = new TensorDesc(adaptiveMultiplier);
			TensorDesc* decay_rates        = new TensorDesc(step);

			graph->addNode((new Operation_Reduce()) //Reduce adaptive Multiplier over spatial dimensions
				->setReduceType(REDUCE_TYPE::MEAN)
				->setInTensor(adaptiveMultiplier)
				->setOutTensor(adaptiveMean)
				->setReduceDimensions({ 2u,3u,4u })
				->setBlendConstants({ ONE, ZERO });
			graph->addNode((new Operation_Fill())  //Fill mean
				->setConstant(adaptiveMean)
				->setOutTensor(adaptiveMeanFilled)
				->setBlendConstants({ ONE, ZERO })
			);
			
			if (decay & WEIGHT_DECAY::L1) { //L1
				graph->addNode((new Operation_Pointwise())
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setInTensor(mem)
					->setOutTensor(decay_rates)
					->setBlendConstants({ this->constants["weightDecay"] , ZERO })
				);
			}
			else { //L2
				graph->addNode((new Operation_Copy())
					->setInTensor(mem)
					->setOutTensor(decay_rates)
					->setBlendConstants({ this->constants["weightDecay"] , ZERO })
				);
			}

			graph->addNode((new Operation_Binary()) //Apply decay
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(decay_rates, adaptiveMeanFilled)
				->setOutTensor(step)
				->setBlendConstants({ ONE, ONE })
			);
		}

		//4.5.: Set stepsize
		if (arg_deltaLr) {
			graph->addNode((new Operation_Binary())
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(oldStepRoot, step)
				->setOutTensor(step)
				->setBlendConstants({ ONE, ZERO })
			);
		}
		else if ((!arg_debias && (arg_momentum != MOMENTUM::NONE || !arg_bounding) ) && arg_momentum != MOMENTUM::NESTROV) { //Else stepSize is inlined in debias factor
			graph->addNode((new Operation_Pointwise())
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_MUL)
				->setInTensor(step)
				->setOutTensor(step)
				->setArgument({ this->constants["stepSize"] })
				->setBlendConstants({ ONE, ZERO })
			);
		}

		//5.: Plain step weight decay
		if ((arg_decay & ~0b11) == WEIGHT_DECAY::STEP_PLAIN) {
			if (decay & WEIGHT_DECAY::L1) { //L1
				graph->addNode((new Operation_Pointwise())
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
			else { //L2
				graph->addNode((new Operation_Copy())
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({ this->constants["weightDecay"], ONE })
				);
			}
		}

		//6.: Regularize step
		//6.1: Clipping
		if (arg_clipping == CLIPPING::STEP) {
			graph->addNode((new Operation_Pointwise())
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_CLIP)
				->setInTensor(step)
				->setOutTensor(step)
				->setArgument({ this->constants["clippingMin"], this->constants["clippingMax"]})
				->setBlendConstants({ ONE, ZERO })
			);
		}

		//6.2.: Projection into tangent space
		if (arg_projection && scaleInvariant) {
			//Compute effective weights
			TensorDesc* mem_squared    = new TensorDesc(mem);
			TensorDesc* mem_normalized = new TensorDesc(mem);
			TensorDesc* l2MemNorm = (new TensorDesc())
				->setTypeId(this->data_type)
				->setSize({ mem->getSize()[0], mem->getSize()[1], 1u, 1u, 1u })
				->setStrides({ mem->getSize()[0] * mem->getSize()[1], mem->getSize()[1], 1u, 1u, 1u })
				->setAlignment(mem->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			TensorDesc* filledL2Norm = new TensorDesc(mem);

			//6.2.1.: Normalize mem
			graph->addNode((new Operation_Pointwise()) //Square mem
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_SQUARE)
				->setInTensor(mem)
				->setOutTensor(mem_squared)
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Reduce()) //Reduce squared mem over spatial dimensions
				->setReduceType(REDUCE_TYPE::SUM)
				->setInTensor(mem_squared)
				->setOutTensor(l2MemNorm)
				->setReduceDimensions({ 2u,3u,4u })
				->setBlendConstants({ ONE, ZERO	})
			);
			graph->addNode((new Operation_Pointwise()) //Root the l2 norm
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_ROOT)
				->setInTensor(l2GradientNorm)
				->setOutTensor(l2GradientNorm)
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Fill()) //Fill norm
				->setConstant(l2GradientNorm)
				->setOutTensor(filledL2Norm)
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Binary())  //Normalize mem
				->setBinaryType(BINARY_TYPE::BINARY_DIV)
				->setInTensors(mem, filledL2Norm)
				->setOutTensor(mem_normalized)
				->setBlendConstants({ ONE, ZERO })
			);

			TensorDesc* dotp_unreduced = new TensorDesc(step);
			TensorDesc* dotp_reduced = new TensorDesc()
				->setTypeId(this->data_type)
				->setSize({ step->getSize[0], 1u, 1u, 1u, 1u })
				->setStrides({ step->getSize[0], 1u, 1u, 1u, 1u })
				->setAlignment(step->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			TensorDesc* dotp_filled = new TensorDesc(step);

			//6.2.2.: Compute dotp
			graph->addNode((new Operation_Binary()) //Compute dotp
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(mem_normalized, step)
				->setOutTensor(dotp_unreduced)
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Reduce()) //Reduce dotp over all spatial dimensions
				->setReduceType(REDUCE_TYPE::SUM)
				->setInTensor(dotp_unreduced)
				->setOutTensor(dotp_reduced)
				->setReduceDimensions({ 2u,3u,4u })
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Fill()) //Fill dotp
				->setBlendConstants(dotp_reduced)
				->setOutTensor(dotp_filled)
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Binary()) //Project
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(dotp_filled, mem_normalized)
				->setOutTensor(step)
				->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0),
					ONE
				})
			);
		}

		//6.3.: Centralization
		if (arg_centralize == CENTRALIZATION::STEP) {
			TensorDesc* step_mean = (new TensorDesc())
				->setTypeId(this->data_type)
				->setSize({ step->getSize()[0], step->getSize()[1], 1u, 1u, 1u })
				->setStrides({ step->getSize()[0] * step->getSize()[1], step->getSize()[1], 1u, 1u, 1u })
				->setAlignment(step->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);

			graph->addNode((new Operation_Reduce()) //Get mean of step for all spatial dimensions
				->setReduceType(REDUCE_TYPE::MEAN)
				->setInTensor(step)
				->setOutTensor(step_mean)
				->setReduceDimensions({ 2u,3u,4u })
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode((new Operation_Fill()) //Fill gradient
				->setConstant(step_mean)
				->setOutTensor(step)
				->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0),
					ONE
				})
			);
		}

		//7.: Update LR buffer
		if (arg_deltaLr) {
			graph->addNode((new Operation_Pointwise())
				->setPointwiseType(POINTWISE_TYPE::SQUARE)
				->setInTensor(step)
				->setOutTensor(optBufs["deltaLr"])
				->setBlendConstants({ this->constants["oldStepSizeNew"], this->constants["oldStepSizeDecay"] })
			);
		}

		//8.: Save debug information
		switch (debug) {
		case DEBUG_INFO::NONE:
			break;
		case DEBUG_INFO::GRADIENT:
			graph->addNode((new Operation_Copy())
				->setInTensor(deltas)
				->setOutTensor(optBufs["debug"])
				->setBlendConstants({ ONE, ZERO })
			);
			break;
		case DEBUG_INFO::STEP:
			graph->addNode((new Operation_Copy())
				->setInTensor(step)
				->setOutTensor(optBufs["debug"])
				->setBlendConstants({ ONE, ZERO })
			); 
			break;
		case DEBUG_INFO::MOMENTUM_BUF:
			graph->addNode((new Operation_Copy())
				->setInTensor(optBuf["momentum")
				->setOutTensor(optBufs["debug"])
				->setBlendConstants({ ONE, ZERO })
			);
			break;
		case DEBUG_INFO::ADAPTIVE_BUF:
			graph->addNode((new Operation_Copy())
				->setInTensor(optBuf["adaptive"])
				->setOutTensor(optBufs["debug"])
				->setBlendConstants({ ONE, ZERO })
			);
			break;
		}

		//9.: Apply step
		graph->addNode((new Operation_Copy())
			->setInTensor(step)
			->setOutTensor(mem)
			->setBlendConstants({
				new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0),
				ONE
			})
		);

		//10.: Lookahead
		if (arg_lookahead > 0) {
			Operation_If* lookahead_condition = (new Operation_If())
				->setConditionConstant(this->constants["lookaheadSyncBool"]);
			
			lookahead_condition->addOperationTrue((new Operation_Binary()) //Update slow weights
				->setBinaryType(BINARY_TYPE::BINARY_SUB)
				->setInTensors(mem, optBufs["lookahead"])
				->setOutTensor(optBuf["lookahead"])
				->setBlendConstants({
					this->constants["lookaheadStepSize"],
					ONE
				})
			);
			lookahead_condition->addOperationTrue((new Operation_Copy()) //Sync fast weights
				->setInTensor(optBufs["lookahead"])
				->setOutTensor(mem)
				->setBlendConstants({ ONE, ZERO })
			);
			graph->addNode(lookahead_condition);
		}
	}
	
	virtual void recalculateGpuConstants() override {
		//1.: Which constants can be inlined?
		bool momentumNeedFactor = ((arg_momentum == MOMENTUM::NESTROV) || (arg_momentum == MOMENTUM::VANILLA && arg_debias));
		bool momentumHasFactor = momentumNeedFactor;

		bool adaptiveNeedFactor = (arg_adaptive != ADAPTIVE::NONE) && (arg_debias != DEBIAS::NONE || arg_rectify != RECTIFY::NONE || arg_momentum == MOMENTUM::NESTROV);
		bool adaptiveNeedInnerFactor = adaptiveNeedFactor && arg_boundAdaptive;
		bool adaptiveHasInnerFactor = adaptiveNeedInnerFactor;
		bool adaptiveHasOuterFactor = !adaptiveNeedInnerFactor && adaptiveNeedFactor && !momentumHasFactor;
		bool adaptiveHasFactor = adaptiveHasInnerFactor || adaptiveHasOuterFactor;
		bool inlineAdaptive = adaptiveNeedFactor && !adaptiveHasFactor;

		bool inlineStepSize = !arg_deltaLr && (momentumHasFactor || adaptiveHasOuterFactor);

		//2.: Update Exponential Moving Average Constants
		if (arg_momentum != MOMENTUM::NONE) {
			this->constants["momentumNew"]     ->setValue(momentumNew  );
			this->constants["momentumDecay"]   ->setValue(momentumDecay);
		}
		if (arg_adaptive != ADAPTIVE::NONE) {
			this->constants["adaptiveNew"]     ->setValue(adaptiveNew  );
			this->constants["adaptiveDecay"]   ->setValue(adaptiveDecay);
		}
		if (arg_deltaLr) {
			this->constants["oldStepSizeNew"]  ->setValue(deltaNew     );
			this->constants["oldStepSizeDecay"]->setValue(deltaDecay   );
		}
		
		//3.: Set weight decay
		switch (arg_decay & ~0b11) {
		case WEIGHT_DECAY::GRAD_PLAIN:
		case WEIGHT_DECAY::STEP_PLAIN:
			this->constants["weightDecay"]->setValue(weightDecay);
			break;
		case WEIGHT_DECAY::GRAD_VANILLA_LR:
			if (arg_deltaLr)
				this->constants["weightDecay"]->setValue(weightDecay);
			else 
				this->constants["weightDecay"]->setValue(weightDecay * stepSize);
			break;
		case WEIGHT_DECAY::STEP_VANILLA_LR:
		case WEIGHT_DECAY::STEP_EFFECTIVE_LR:
		case WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN:
			if (inlineStepSize)
				this->constants["weightDecay"]->setValue(weightDecay * stepSize);
			else
				this->constants["weightDecay"]->setValue(weightDecay);
		}

		//4.: Set clipping and bounding
		if (arg_clipping) {
			this->constants["clippingMin"]->setValue(clippingMin);
			this->constants["clippingMin"]->setValue(clippingMax);
		}
		if (arg_boundAdaptive) {
			this->constants["adaptiveBoundMin"]->setValue(boundMin);
			this->constants["adaptiveBoundMin"]->setValue(boundMax);
		}

		//5.: Set debiasing constants (these might not all be needed but those that are will be set to the right value)
		T momentumFac = (T)1;
		T nestrovFac  = (T)1;
		T adaptiveFac = (T)1;

		//5.1.: Compute adaptive factor
		if (adaptiveNeedFactor) { //Own factor
			switch (arg_debias) {
			case DEBIAS::NONE:
				break;
			case DEBIAS::APPROX:
				adaptiveFac /= (T)1 - pow(adaptiveDecay, timeStep);
				break;
			case DEBIAS::EXACT:
				static T adaprod = (T)1;
				adaprod *= adaptiveDecay;
				adaptiveFac /= (T)1 - adaprod;
				break;
			}

			//TODO: Maybe need to divide by factor instead of multiplying
			switch (arg_rectify) {
			case RECTIFY::NONE:
				break;
			case RECTIFY::APPROX:
				T rho_inf = (T)2 / adaptiveNew - (T)1;
				T rho_t = rho_inf - (T)2 * timeStep * pow(adaptiveDecay, timeStep) / ((T)1 - pow(adaptiveDecay, timeStep));
				this->constants["rectificationBool"]->setValue(rho_t > (T)4);
				adaptiveFac *= ((rho_inf - (T)4) * (rhi_inf - (T)2) * rho_t) / ((rho_t - (T)4) * (rho_t - (T)2) * (rho_inf));
				break;
			case RECTIFY::EXACT:
				T rho_inf = (T)2 / adaptiveNew - (T)1;
				static T adaprod = (T)1;
				adaprod *= adaptiveDecay;
				T rho_t = rho_inf - (T)2 * timeStep * adaprod / ((T)1 - adaprod);
				this->constants["rectificationBool"]->setValue(rho_t > (T)4);
				adaptiveFac *= ((rho_inf - (T)4) * (rhi_inf - (T)2) * rho_t) / ((rho_t - (T)4) * (rho_t - (T)2) * (rho_inf));
				break;
			}

			if (arg_momentum == MOMENTUM::NESTROV)
				adaptiveFac *= adaptiveDecay;
		}
		if (adaptiveHasOuterFactor && inlineStepSize) //Inline step size
			adaptiveFac *= stepSize;

		//5.2.: Compute momentum factors
		//Own factor
		switch (arg_momentum) {
		case MOMENTUM::NONE:
			break;
		case MOMENTUM::NESTROV:
			momentumFac *= momentumDecayNext;
			nestrovFac *= momentumNew;

			switch (arg_debias) {
			case DEBIAS::NONE:
				break;
			case DEBIAS::APPROX:
				T tmp = pow(momentumDecay, timeStep);
				momentumFac /= (T)1 - tmp * momentumDecay;
				nestrovFac  /= (T)1 - tmp;
				break;
			case DEBIAS::EXACT:
				static T momProd = (T)1;
				momProd *= momentumDecay;
				momentumFac /= (T)1 - momProd * momentumDecayNext;
				nestrovFac  /= (T)1 - momProd;
				break;
			}
		case MOMENTUM::VANILLA:
			switch (arg_debias) {
			case DEBIAS::NONE:
				break;
			case DEBIAS::APPROX:
				momentumFac /= (T)1 - pow(momentumDecay, timeStep);
				break;
			case DEBIAS::EXACT:
				static T momProd = (T)1;
				momProd *= momentumDecay;
				momentumFac /= (T)1 - momProd;
				break;
			}
		}

		if (inlineAdaptive) { //Inline adpative
			switch (arg_adaptive) {
			case ADAPTIVE::SQRT:
				momentumFac *= pow(adaptiveFac, -1./2.);
				nestrovFac  *= pow(adaptiveFac, -1./2.);
				break;
			case ADAPTIVE:CBRT:
				momentumFac *= pow(adaptiveFac, -1./3.);
				nestrovFac  *= pow(adaptiveFac, -1./3.);
				break;
			}
		}

		if (inlineStepSize) { //Inline step size
			momentumFac *= stepSize;
			nestrovFac  *= stepSize;
		}

		//5.3.: Set factors
		if (momentumHasFactor)
			this->constants["momentumDebias"]->setValue(momentumFac);
		if (arg_momentum == MOMENTUM::NESTROV)
			this->constants["nestrovFactor"]->setValue(nestrovFac);
		if (adaptiveHasFactor)
			this->constants["adaptiveDebias"]->setValue(adaptiveFac);
		if (!inlineStepSize && !arg_deltaLr) 
			this->constants["stepSize"]->setValue(stepSize);
	
		//6.: Epsilon
		if (adaptive != ADAPTIVE::NONE)
			this->constants["epsilon"]->setValue(epsilon);

		//7.: Lookahead
		if (arg_lookahead > 0u) {
			static stepsUntilSync = arg_lookahead;
			if (stepsUntilSync == 0) {
				stepsUntilSync = arg_lookahead;
				this->constants["lookaheadSyncBool"].setValue(1);
			}
			else {
				this->constants["lookaheadSyncBool"].setValue(0);
				stepsUtilSync--;
			}
		}
	}
};














template<
	MOMENTUM momentum         = MOMENTUM::NESTROV,
	ADAPTIVE adaptive         = ADAPTIVE::NONE,
	WEIGHT_DECAY decay        = WEIGHT_DECAY::L2 ^ WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN,
	CLIPPING clipping         = CLIPPING::GRADIENT, 
	DEBIAS de_bias            = DEBIAS::NONE, 
	CENTRALIZATION centralize = CENTRALIZATION::GRADIENT,
	uint32_t lookahead        = 0u,
	bool delta_lr             = false,
	bool projection           = true,
	DEBUG_INFO debug          = DEBUG_INFO::NONE
>
class FirstOrder_Optimizer : public Optimizer {
	//constants (all indirection)
	constexpr uint32_t stepNumber_idx        = 0;                                                       //The number of optimization step
	constexpr uint32_t stepSize_idx          = stepNumber_idx        + (debias != DEBIAS::NONE);        //The learning rate / step size
	constexpr uint32_t epsilon_idx           = stepSize_idx          + (!delta_lr);                     //Small constant for numeric stability
	constexpr uint32_t momentumDecay_idx     = epsilon_idx           + 1;                               //m_{n+1} = (momentumDecay) * m_n + (momentumNew) * g_n
	constexpr uint32_t momentumDecayNext_idx = momentumDecay_idx     + (momentum != MOMENTUM::NONE);    //The momentumDecay of the next timestep
	constexpr uint32_t momentumNew_idx       = momentumDecayNext_idx + (momentum != MOMENTUM::NESTROV); //m_{n+1} = (momentumDecay) * m_n + (momentumNew) * g_n
	constexpr uint32_t momentumNewNext_idx   = momentumNew_idx       + (momentum != MOMENTUM::NONE);    //The momentumNew of the next timestep
	constexpr uint32_t momentumDebias_idx    = momentumNewNext_idx   + (momentum != MOMENTUM::NESTROV); //The factor needed to debias momentum. If adaptive is also activated and weight decay is not EFFECTIVE_LR_MEAN, this factor can be fused with adaptiveDebias
	constexpr uint32_t momentumNestrov_idx   = momentumDebias_idx    + (momentum != MOMENTUM::NONE && 
		debias != DEBIAS::NONE);	                                                                    //momentumNex/1-momentumDecay^t, used for nestrov momentum
	constexpr uint32_t adaptiveDecay_idx     = momentumNestrov_idx   + (momentum == MOMENTUM::NESTROV); //a_{n+1} = (adaptiveDecay) * a_n + (adaptiveNew) * (g_n)^2
	constexpr uint32_t adaptiveNew_idx       = adaptiveDecay_idx     + (adaptive != ADAPTIVE::NONE);    //a_{n+1} = (adaptiveDecay) * a_n + (adaptiveNew) * (g_n)^2
	constexpr uint32_t adaptiveDebias_idx    = adaptiveNew_idx       + (adaptive != ADAPTIVE::NONE);    //The factor needed to debias adaptive. If momentum is also activated and weight decay os not EFFECTIVE_LR_MEAN, this factor is not needed as it is fused with momentumDebias
	constexpr uint32_t oldStepsDecay_idx     = adaptiveDebias_idx    + (adaptive != ADAPTIVE::NONE && 
		debias != DEBIAS::NONE && (momemtum == MOMENTUM::NONE || decay == WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN));        //s_{n+1} = (oldStepsDecay) * s_n + (oldStepsNew) * step_n
	constexpr uint32_t oldStepNew_idx        = oldStepsDecay_idx     + (delta_lr);                      //s_{n+1} = (oldStepsDecay) * s_n + (oldStepsNew) * step_n
	constexpr uint32_t lookaheadSyncBool_idx = oldStepNew_idx        + (delta_lr);                      //1 if synching and slow weight update has to occur, 0 otherwise
	constexpr uint32_t lookaheadStepSize_idx = lookaheadSyncBool_idx + (lookahead > 0u);                //LR for slow weight synchronization. Negative.
	constexpr uint32_t clippingMin_idx       = lookaheadStepSize_idx + (lookahead > 0u);                //The minimum value to clip gradient/step to
	constexpr uint32_t clippingMax_idx       = clippingMin           + (clipping != CLIPPING::NONE);    //The maximum value to clip gradient/step to
	constexpr uint32_t decayRate_idx         = clippingMax_idx       + (clipping != CLIPPING::NONE);    //The weight decay rate, already multiplied with lr if GRAD_VANILLA_LR. Negative if STEP_PLAIN. (w -= decayRateNeg * f(w), f(x)=x oder f(x)=sgn(x))

	//optBufs
	constexpr uint32_t momentum_idx  = 0;
	constexpr uint32_t adaptive_idx  = momentum_idx  + (momentum != MOMENTUM::NONE);
	constexpr uint32_t lookahead_idx = adaptive_idx  + (adaptive != ADAPTIVE::NONE); //The slow weights
	constexpr uint32_t delta_idx     = lookahead_idx + (lookahead > 0u            );
	constexpr uint32_t debug_idx     = delta_idx     + (delta_idx);

public:
	FirstOrder_Optimizer(TYPE data_type, TYPE computation_type)
	{
		//0.: Check consistency
		if (debug == DEBUG_INFO::MOMENTUM_BUF && momentum == MOMENTUM::NONE) {
			fprintf(stderr, "[ERROR] Optimizer can't save momentum buffer if no momentum is used! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}
		if (debug == DEBUG_INFO::ADAPTIVE_BUF && adaptive == ADAPTIVE::NONE) {
			fprintf(stderr, "[ERROR] Optimizer can't save adaptive buffer if no adaptivity is used! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}
		

		//TODO
		this->data_type = data_type;
		this->computation_type = computation_type;

		this->optBuffers = std::vector<TensorDesc*>();
		this->constants = std::vector<ConstantDesc*>();

		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true));   //-Alpha
	}

	//TODO: Centralize/decay mean over C-dimension?
	//TODO: Decay EFFECTIVE_LR_MEAN uses mean(sqrt(v)) instead of sqrt(mean(v)) which is not equivalent!
	//TODO: Fuse bias and nestrov
	//TODO: line search
	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas, bool scaleInvariant = false) override {
		//0.: Check input
		if (this->data_type != mem->getTypeId())    throw new std::runtime_error("[ERROR] Optimizer requires memory to be of the previously specified data type");
		if (this->data_type != deltas->getTypeId()) throw new std::runtime_error("[ERROR] Optimizer requires deltas to be of the previously specified data type");

		//1.: Get associated buffers
		auto associatedBufs_p = this->optBuffers.find(mem);
		std::vector<TensorDesc*> associatedBufs;
		if (associatedBufs_p == this->optBuffers.end()) {
			associatedBufs = std::vector<TensorDesc*>();
			if (momentum != MOMENTUM::NONE)	associatedBufs.push_back(new TensorDesc(mem)->setIsNeeded(true));
			if (adaptive != ADAPTIVE::NONE) associatedBufs.push_back(new TensorDesc(mem)->setIsNeeded(true));
			if (lookahead                 ) associatedBufs.push_back(new TensorDesc(mem)->setIsNeeded(true));
			if (delta_lr                  ) associatedBufs.push_back(new TensorDesc(mem)->setIsNeeded(true));
			if (debug                     ) associatedBufs.push_back(new TensorDesc(mem)->setIsNeeded(true));
			this->optBuffers[mem] = associatedBufs;
		}
		else
			associatedBufs = associatedBufs_p->second;

		//2.: Regularization pre update
		//2.1: Clipping
		if constexpr (clipping == CLIPPING::GRADIENT) {
			Operation_Pointwise* clip = new Operation_Pointwise()
				->setInTensor(deltas)
				->setOutTensor(deltas)
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_CLIP)
				->setArgument({ this->constants[clippingMin_idx], this->constants[clippingMax_idx] })
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(clip);
		}

		//2.2.: Weight decay
		if constexpr ((decay >> 2) == WEIGHT_DECAY::GRAD_PLAIN || (decay >> 2) == WEIGHT_DECAY::GRAD_VANILLA_LR) {
			if constexpr (decay & WEIGHT_DECAY::L1) {
				Operation_Pointwise* l1_reg = new Operation_Pointwise()
					->setInTensor(mem)
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setOutTensor(delta)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l1_reg);
			}
			else if constexpr (decay & WEIGHT_DECAY::L2) {
				Operation_Copy* l2_reg = new Operation_Copy()
					->setInTensor(mem)
					->setOutTensor(delta)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l2_reg);
			}
			else {
				fprintf(stderr, "[ERROR] If weight decay is used, it must be either L1 or L2! (File %s, Line %d)\n", __FILE__, __LINE__);
				std::exit(-1);
			}
		}

		if constexpr ((decay >> 2) == WEIGHT_DECAY::STEP_PLAIN) {
			if constexpr (decay & WEIGHT_DECAY::L1) {
				Operation_Pointwise* l1_reg = new Operation_Pointwise()
					->setInTensor(mem)
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setOutTensor(mem)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l1_reg);
			}
			else if constexpr (decay & WEIGHT_DECAY::L2) {
				Operation_Copy* l2_reg = new Operation_Copy()
					->setInTensor(mem)
					->setOutTensor(mem)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l2_reg);
			}
			else {
				fprintf(stderr, "[ERROR] If weight decay is used, it must be either L1 or L2! (File %s, Line %d)\n", __FILE__, __LINE__);
				std::exit(-1);
			}
		}

		//2.3.: Centralization
		if constexpr (centralize == CENTRALIZATION::GRADIENT) {
			TensorDesc* gradient_mean = new TensorDesc()
				->setTypeId(this->data_type)
				->setSize({ deltas->getSize[0], 1u, 1u, 1u, 1u })
				->setStrides({ deltas->getSize[0], 1u, 1u, 1u, 1u })
				->setAlignment(deltas->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			Operation_Reduce* computeMean = new Operation_Reduce()
				->setInTensor(deltas)
				->setOutTensor(gradient_mean)
				->setReduceDimensions({ 2u,3u,4u }) //All spacial dimension
				->setReduceType(REDUCE_TYPE::MEAN)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			Operation_Bias* gradientShift = new Operation_Bias()
				->setInTensors(deltas, gradient_mean)
				->setOutTensor(deltas)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
					});
			graph->addNode(computeMean);
			graph->addNode(gradientShift);
		}

		//3.: Update gradient buffers
		//3.1.: Momentum buffer
		Operation_Copy* momentumUpd;
		if constexpr (momentum != Momentum::NONE){
			Operation_Copy momentumUpd = new Operation_Copy()
				->setInTensor(deltas)
				->setOutTensor(associatedBufs[momentum_idx])
				->setBlendConstants({ this->constants[momentumNew_idx], this->constants[momentumDecay_idx] });
			graph->addNode(momentumUpd);
			break;
		}

		//3.2.: Adaptive buffer
		if constexpr (adaptive != ADAPTIVE::NONE) {
			Operation_Pointwise* gradientSquare = new Operation_Pointwise()
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_SQUARE)
				->setInTensor(deltas)
				->setOutTensor(associatedBufs[adaptive_idx])
				->setBlendConstants({ this->constants[adaptiveNew_idx], this->constants[adaptiveDecay_idx] });
			graph->addNode(gradient)
			break;
		}
		
		//4.: Debias
		//If no debiasing is occuring, just copy tensor over.
		TensorDesc* momentum_debiased;
		TensorDesc* adaptive_debiased; 
		if constexpr (momentum != MOMENTUM::NONE) momentum_debiased = new TensorDesc(associatedBufs[momentum_idx]);
		if constexpr (adaptive != ADAPTIVE::NONE) adaptive_debiased = new TensorDesc(associatedBufs[adaptive_idx]);
		
		if constexpr (debias == DEBIAS::NONE) {
			//Move momentum
			if (momentum != MOMENTUM::NONE) {
				Operation_Copy* moveMomentum = new Operation_Copy()
					->setInTensor(associatedBufs[momentum_idx])
					->setOutTensor(momentum_debiased)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(moveMomentum);
			}

			//Move adaptive
			if (adaptive != ADAPTIVE::NONE) {
				Operation_Copy* moveAdaptive = new Operation_Copy()
					->setInTensor(associatedBufs[adaptive_idx])
					->setOutTensor(adaptive_debiased)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(moveAdaptive);
			}
		}
		else {
			//Momentum
			if (momentum != MOMENTUM::NONE) {
				Operation_Pointwise debiasMometum = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_MUL)
					->setInTensor(associatedBufs[momentum_idx])
					->setOutTensor(momentum_debiased)
					->setArgument(this->constants[momentumDebias_idx])
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(debiasMometum);
			}

			//Adaptive
			if (adaptive != ADAPTIVE::NONE) {
				if (momentum != MOMENTUM::NONE && decay != WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN) {
					//Operation is not needed, as it is fused with momentum debias
					Operation_Copy* moveAdaptive = new Operation_Copy()
						->setInTensor(associatedBufs[adaptive_idx])
						->setOutTensor(adaptive_debiased)
						->setBlendConstants({
								new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
								new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
							});
					graph->addNode(moveAdaptive);
				}
				else {
					//Debiasing needed
					Operation_Pointwise debiasAdaptive = new Operation_Pointwise()
						->setPointwiseType(POINTWISE_TYPE::POINTWISE_MUL)
						->setInTensor(associatedBufs[adaptive_idx])
						->setOutTensor(adaptive_debiased)
						->setArgument(this->constants[adaptiveDebias_idx])
						->setBlendConstants({
								new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
								new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
							});
					graph->addNode(debiasAdaptive);
				}
			}
		}
		
		//5.: Build step buffer
		TensorDesc* step = new TensorDesc(mem); //This holds the step every weights is moved in (actually this is the value that will be subtracted from every weight)

		//5.1: Set gradient/momentum
		switch (momentum) {
		case MOMENTUM::NONE:
			Operation_Binary* setGradient = new Operation_Binary()
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(step, deltas)
				->setOutTensor(step)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(setGradient);
			break;

		case MOMENTUM::NESTROV:
			//TODO: Fuse with debias
			Operation_Copy* nestrovGradient = new Operation_Copy()
				->setInTensor(deltas)
				->setOutTensor(momentum_debiased)
				->setBlendConstants({
						this->constants[momentumNestrov_idx],
						this->constants[momentumDecayNext_idx]
					});
			Operation_Pointwise* nestrovAdaptive = new Operation_Pointwise()
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_MUL)
				->setInTensor(adaptive_debiased)
				->setArgument({ this->constants[momentumNewNext_idx] })
				->setOutTensor(adaptive_debiased)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(nestrovGradient);
			graph->addNode(nestrovAdaptive);
		case MOMENTUM::VANILLA:
			Operation_Binary* setMomentum = new Operation_Binary()
				->setInTensors(step, momentum_debiased)
				->setOutTensor(step)
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(setMomentum);

			break;
		}
		
		//5.2.: Effective lr decay
		if constexpr ((decay >> 2) == WEIGHT_DECAY::STEP_EFFECTIVE_LR) {			
			if constexpr (decay & WEIGHT_DECAY::L1) {
				Operation_Pointwise* l1_reg = new Operation_Pointwise()
					->setInTensor(mem)
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setOutTensor(step)
					->setBlendConstants({
							-this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l1_reg);
			}
			else if constexpr (decay & WEIGHT_DECAY::L2) {
				Operation_Copy* l2_reg = new Operation_Copy()
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l2_reg);
			}
			else {
				fprintf(stderr, "[ERROR] If weight decay is used, it must be either L1 or L2! (File %s, Line %d)\n", __FILE__, __LINE__);
				std::exit(-1);
			}
		}
		
		//5.3.: Compute adaptive lr
		TensorDesc* adaptiveMultiplier  = new TensorDesc(adaptive_debiased);
		switch (adaptive) {
		case ADAPTIVE::NONE:
			break;
		case ADAPTIVE::SQRT:
			{
				Operation_Pointwise* addEpsilon = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_ADD)
					->setInTensor(adaptive_debiased)
					->setOutTensor(adaptive_debiased)
					->setArgument(this->constants[epsilon_idx])
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Pointwise* adaptivelr = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_ROOT_REC)
					->setInTensor(adaptive_debiased)
					->setOutTensor(adaptiveMultiplier)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Binary* mulAdaptive = new Operation_Binary()
					->setBinaryTypesetBinaryType(BINARY_TYPE::BINARY_MUL)
					->setInTensors(adaptiveMultiplier, step)
					->setOutTensor(step)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(addEpsilon);
				graph->addNode(adaptivelr);
				graph->addNode(mulAdaptive);
			}
			break;
		case ADAPTIVE::CBRT:
			{
				Operation_Pointwise* addEpsilon = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_ADD)
					->setInTensor(adaptive_debiased)
					->setOutTensor(adaptive_debiased)
					->setArgument(this->constants[epsilon_idx])
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
			
				Operation_Pointwise* adaptivelr = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_POW)
					->setInTensor(adaptive_debiased)
					->setOutTensor(adaptiveMultiplier)
					->setArgument({ new ConstantDesc()->setUseIndirection(false)->setValue(-1.0 / 3.0) })
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Binary* mulAdaptive = new Operation_Binary()
					->setBinaryType(BINARY_TYPE::BINARY_MUL)
					->setInTensors(adaptiveMultiplier, step)
					->setOutTensor(step)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(addEpsilon);
				graph->addNode(adaptivelr);
				graph->addNode(mulAdaptive);
			}
			break;
		}

		//5.4.: Effective lr mean & vanilla lr decay
		if constexpr ((decay >> 2) == WEIGHT_DECAY::STEP_VANILLA_LR || 
			((decay >> 2) == WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN && adaptive == ADAPTIVE::NONE)) {
			if constexpr (decay & WEIGHT_DECAY::L1) {
				Operation_Pointwise* l1_reg = new Operation_Pointwise()
					->setInTensor(mem)
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setOutTensor(step)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l1_reg);
			}
			else if constexpr (decay & WEIGHT_DECAY::L2) {
				Operation_Copy* l2_reg = new Operation_Copy()
					->setInTensor(mem)
					->setOutTensor(step)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
						});
				graph->addNode(l2_reg);
			}
			else {
				fprintf(stderr, "[ERROR] If weight decay is used, it must be either L1 or L2! (File %s, Line %d)\n", __FILE__, __LINE__);
				std::exit(-1);
			}
		}
		else if constexpr ((decay >> 2) == WEIGHT_DECAY::STEP_EFFECTIVE_LR_MEAN) {
			TensorDesc* adaptiveMean = new TensorDesc()
				->setTypeId(this->data_type)
				->setSize({ adaptiveMultiplier->getSize[0], 1u, 1u, 1u, 1u })
				->setStrides({ adaptiveMultiplier->getSize[0], 1u, 1u, 1u, 1u })
				->setAlignment(adaptiveMultiplier->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			TensorDesc* adaptiveMeanFilled = new TensorDesc(adaptiveMultiplier);
			TensorDesc* decay_rates        = new TensorDesc(step);

			Operation_Reduce* computeMean = new Operation_Reduce()
				->setInTensor(adaptiveMultiplier)
				->setOutTensor(adaptiveMean)
				->setReduceDimensions({ 2u,3u,4u }) //All spatial dimensions
				->setReduceType(REDUCE_TYPE::MEAN)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			Operation_Fill* fillMean = new Operation_Fill()
				->setConstant(adaptiveMean)
				->setOutTensor(adaptiveMeanFilled)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0),
					});
			graph->addNode(computeMean);
			graph->addNode(fillMean);

			if constexpr (decay & WEIGHT_DECAY::L1) {
				Operation_Pointwise* l1_reg = new Operation_Pointwise()
					->setInTensor(mem)
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SIGN)
					->setOutTensor(decay_rates)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(l1_reg);
			}
			else if constexpr (decay & WEIGHT_DECAY::L2) {
				Operation_Copy* l2_reg = new Operation_Copy()
					->setInTensor(mem)
					->setOutTensor(decay_rates)
					->setBlendConstants({
							this->constants[decayRate_idx],
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				graph->addNode(l2_reg);
			}
			else {
				fprintf(stderr, "[ERROR] If weight decay is used, it must be either L1 or L2! (File %s, Line %d)\n", __FILE__, __LINE__);
				std::exit(-1);
			}

			Operation_Binary* applyDecay = new Operation_Binary()
				->setBinaryType(BINARY_TYPE::BINARY_MUL)
				->setInTensors(decay_rates, adaptiveMeanFilled)
				->setOutTensor(step)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
					});
			graph->addNode(applyDecay);
		}

		//5.5.: Set stepsize
		if constexpr (delta_lr) {
			Operation_Copy* setLr = new Operation_Copy()
				->setInTensor(associatedBufs[delta_idx])
				->setOutTensor(step)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(setLr);
		}
		else {
			Operation_Fill* setLr = new Operation_Fill()
				->setConstant(this->constants[stepSize_idx])
				->setOutTensor(step)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(setLr);
		}
		
		//6.: Regularize step
		//6.1: Clipping
		if constexpr (clipping == CLIPPING::STEP) {
			Operation_Pointwise* clip = new Operation_Pointwise()
				->setPointwiseType(POINTWISE_TYPE::POINTWISE_CLIP)
				->setInTensor(step)
				->setOutTensor(step)
				->setArgument({ this->constants[clippingMin_idx], this->constants[clippingMax_idx] })
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(clip);
		}

		//6.2.: Projection into tangent space
		if constexpr (projection) {
			if (scaleInvariant) {
				//Compute effective weights
				TensorDesc* mem_squared    = new TensorDesc(mem);
				TensorDesc* mem_normalized = new TensorDesc(mem);
				TensorDesc* l2GradientNorm = new TensorDesc()
					->setTypeId(this->data_type)
					->setSize({ mem->getSize[0], 1u, 1u, 1u, 1u })
					->setStrides({ mem->getSize[0], 1u, 1u, 1u, 1u })
					->setAlignment(mem->getAlignment())
					->setIsNeeded(false)
					->setUseIndirection(false);
				TensorDesc* filledL2Norm = new TensorDesc(mem);
			
				Operation_Pointwise* weightSquaring = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_SQUARE)
					->setInTensor(mem)
					->setOutTensor(mem_squared)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Reduce* reduceSquaredGradients = new Operation_Reduce()
					->setInTensor(mem_squared)
					->setOutTensor(l2GradientNorm)
					->setReduceDimensions({ 2u,3u,4u }) //All spatial dimension
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Pointwise* rootNorm = new Operation_Pointwise()
					->setPointwiseType(POINTWISE_TYPE::POINTWISE_ROOT)
					->setInTensor(l2GradientNorm)
					->setOutTensor(l2GradientNorm)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Fill* fillNorm = new Operation_Fill()
					->setConstant(l2GradientNorm)
					->setOutTensor(filledL2Norm)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Binary* normalizeWeights = new Operation_Binary()
					->setBinaryType(BINARY_TYPE::BINARY_DIV)
					->setInTensors(mem, fillNorm)
					->setOutTensor(mem_normalized)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});

				graph->addNode(weightSquaring);
				graph->addNode(reduceSquaredGradients);
				graph->addNode(rootNorm);
				graph->addNode(fillNorm);
				graph->addNode(normalizeWeights);

				TensorDesc* dotp_unreduced = new TensorDesc(step);
				TensorDesc* dotp_reduced   = new TensorDesc()
					->setTypeId(this->data_type)
					->setSize({ step->getSize[0], 1u, 1u, 1u, 1u })
					->setStrides({ step->getSize[0], 1u, 1u, 1u, 1u })
					->setAlignment(step->getAlignment())
					->setIsNeeded(false)
					->setUseIndirection(false);
				TensorDesc* dotp_filled    = new TensorDesc(step);

				Operation_Binary* computeDotp = new Operation_Binary()
					->setBinaryType(BINARY_TYPE::BINARY_MUL)
					->setInTensors(mem_normalized, step)
					->setOutTensor(dotp_unreduced)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Reduce* reduceDotp = new Operation_Reduce()
					->setReduceType(REDUCE_TYPE::SUM)
					->setInTensor(dotp_unreduced)
					->setOutTensor(dotp_reduced)
					->setReduceDimensions({ 2u,3u,4u }) //All spatial dimensions
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Fill* fillDotp = new Operation_Fill()
					->setBlendConstants(dotp_reduced)
					->setOutTensor(dotp_filled)
					->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
						});
				Operation_Binary* projectStep = new Operation_Binary()
					->setBinaryType(BINARY_TYPE::BINARY_MUL)
					->setInTensors(dotp_filled, mem_normalized)
					->setOutTensor(step)
					->setBlendConstants({
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
							new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0)
						});
				graph->addNode(computeDotp);
				graph->addNode(reduceDotp);
				graph->addNode(fillDotp);
				graph->addNode(projectStep);
			}
		}

		//6.3.: Centralization
		if constexpr (centralize == CENTRALIZATION::STEP) {
			TensorDesc* step_mean = new TensorDesc()
				->setTypeId(this->data_type)
				->setSize({ step->getSize[0], 1u, 1u, 1u, 1u })
				->setStrides({ step->getSize[0], 1u, 1u, 1u, 1u })
				->setAlignment(step->getAlignment())
				->setIsNeeded(false)
				->setUseIndirection(false);
			Operation_Reduce* computeMean = new Operation_Reduce()
				->setInTensor(step)
				->setOutTensor(step_mean)
				->setReduceDimensions({ 2u,3u,4u }) //All spatial dimensions
				->setReduceType(REDUCE_TYPE::MEAN)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			Operation_Fill* stepShift = new Operation_Fill()
				->setConstant(step_mean)
				->setOutTensor(step)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
					});
			graph->addNode(computeMean);
			graph->addNode(stepShift);
		}

		//7.: Apply step
		Operation_Copy* applyStep = new Operation_Copy()
			->setInTensor(step)
			->setOutTensor(mem)
			->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(-1.0),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
				});
		graph->addNode(applyStep);

		//8.: Save debug information
		switch (debug) {
		case DEBUG_INFO::NONE:
			break;
		case DEBUG_INFO::GRADIENT:
			Operation_Copy* saveGrad = new Operation_Copy()
				->setInTensor(deltas)
				->setOutTensor(associatedBufs[debug_idx])
				->setBlendConstants({
						new ConstantDesc()->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc()->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(saveGrad);
			break;
		case DEBUG_INFO::STEP:
			Operation_Copy* saveStep = new Operation_Copy()
				->setInTensor(step)
				->setOutTensor(associatedBufs[debug_idx])
				->setBlendConstants({
						new ConstantDesc()->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc()->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(saveStep);
			break;
		case DEBUG_INFO::MOMENTUM_BUF:
			Operation_Copy* saveMomentum = new Operation_Copy()
				->setInTensor(associatedBufs[momentum_idx)
				->setOutTensor(associatedBufs[debug_idx])
				->setBlendConstants({
						new ConstantDesc()->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc()->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(saveMomentum);
			break;
		case DEBUG_INFO::ADAPTIVE_BUF:
			Operation_Copy* saveAdaptive = new Operation_Copy()
				->setInTensor(associatedBufs[adaptive_idx])
				->setOutTensor(associatedBufs[debug_idx])
				->setBlendConstants({
						new ConstantDesc()->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc()->setUseIndirection(false)->setValue(0.0)
					});
			graph->addNode(saveAdaptive);
			break;
		}

		//9.: Lookahead
		if constexpr (lookahead > 0) {
			Operation_If* lookahead_condition = new Operation_If()
				->setConditionConstant(this->constants[lookaheadSyncBool_idx]);
			Operation_Binary* updateSlow = new Operation_Binary()
				->setBinaryType(BINARY_TYPE::BINARY_SUB)
				->setInTensors(mem, associatedBufs[lookahead_idx])
				->setOutTensor(associtedBufs[lookahead_idx)
				->setBlendConstants({
						this->constants[lookaheadStepSize_idx],
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
					});
			Operation_Copy* updateFast = new Operation_Copy()
				->setInTensor(associatedBufs[lookahead_idx])
				->setOutTensor(mem)
				->setBlendConstants({
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
						new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
					});
			lookahead_condition->addOperationTrue(updateSlow);
			lookahead_condition->addOperationTrue(updateFast);
			graph->addNode(lookahead_condition);
		}
			
		//10.: Update LR buffer
		if (delta_lr) {
			Operation_Copy* deltaLr_upd = new Operation_Copy()
				->setInTensor(step)
				->setOutTensor(associatedBufs[delta_idx])
				->setBlendConstants({ this->constants[oldStepNew_idx], this->constants[oldStepsDecay_idx] });
			graph->addNode(deltaLr_upd);
		}
	}

	virtual void initMem() override { /*TODO*/ }
	virtual void setLearningRates(double alpha, double beta1 = -1., double beta2 = -1.) override {
		this->constants[0]->setValue(-alpha);
	
	
		//debiasing is hard for nestrov 1-beta^(t+1).
	}

	virtual OPTIMIZER_TYPE getType() const override {
		return
	}
};




class Zero_Order_Optimizer {}; //Gridsearch, approx gradient, ...
class Quasi_Newton_Optimizer{};   //BFGS, adahession, newton, newton-raphson, cubic with old point and gradient

//Conjugate gradients

/*
 Optimizers
 ----------
 p = paramter to optimize
 p_0 = intial value of p
 alpha = lr(step size)
 beta1 = momentum decay rate
 beta2 = adaptivity rate
 beta3 = delta buffer decay rate
 eps = epsilon that avoid division by zero
 g = gradient
 u = gradient buffer
 v = gradient squared buffer
 u_deb = debiased gradiant buffer
 v_deb = debiased gradient squared buffer
 w = delta buffer
 t = time step
 tmp = temporary value

 SGD:
 p = p - alpha * g

 SGD with momentum:
 u = beta1 * u + (1 - beta1) * g
 p = p - alpha * u

 Adagrad:
 v = v + g*g
 p = p - alpha/sqrt(v + eps) * g

 RMSProp:
 v = beta2 * v + (1 - beta2) * (g*g)
 p = p - alpha/sqrt(v + eps) * g

 Adadelta:
 v = beta2 * v + (1 - beta2) * (g*g)
 tmp = sqrt(w + eps)/sqrt(v + eps) * g
 p = p - tmp
 w = beta3 * w + (1 - beta3) * (tmp*tmp)

 Adam:
 u = beta1 * u + (1 - beta1) * g
 v = beta2 * v + (1 - beta2) * (g*g)
 u_deb = u / (1 - beta1^t)
 v_deb = v / (1 - beta2^t)
 p = p - alpha/(sqrt(v_deb + eps) * u_deb

 AdaMax:
 u = beta1 * u + (1 - beta1) * g
 v = max(beta2 * v, |g|)
 u_deb = u / (1 - beta1^t)
 p = p - alpha/v * u_deb

 Nadam:
 u = beta1 * u + (1 - beta1) * g
 v = beta2 * v + (1 - beta2) * (g*g)
 u_deb = u / (1 - beta1^t)
 v_deb = v / (1 - beta2^t)
 p = p - alpha/(sqrt(v_deb + eps) * (beta1 * u_deb + (1 - beta1)/(1 - beta1^t) * g)

 AMSGrad:
 u = beta1 * u + (1 - beta1) * g
 v = beta2 * v + (1 - beta2) * (g*g)
 v_deb = max(v_deb, v)
 p = p - alpha/(sqrt(v_deb + eps) * u_deb

 Madgrad:
 u = u + alpha * sqrt(t + 1) * g
 v = v + alpha * sqrt(t + 1) * (g*g)
 p = beta1 * p + (1 - beta1) * (p_0 - 1/cbrt(v + eps) * u)

 AdaBound:
 u = beta1 * u + (1 - beta1) * g
 v = beta2 * v + (1 - beta2) * (g*g)
 tmp = clamp(alpha/sqrt(v), lower_bound(t), upper_bound(t))/sqrt(t)
 p = p - tmp * u

 AdaMod:
 u = beta1 * u + (1 - beta1) * g
 v = beta2 * v + (1 - beta2) * (g*g)
 u_deb = u / (1 - beta1^t)
 v_deb = v / (1 - beta2^t)
 tmp = alpha / sqrt(v_deb + eps)
 w = beta3 * w + (1 - beta3) * tmp
 p = p - min(w, tmp) * u_deb
 
 DiffGrad:

 AdamP:

 AdaBelieve:

 Ranger:

 DeepMemory:

*/




















//Old






class Optimizer_None : public Optimizer {
public:
	Optimizer_None() = default;

	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas) override { /*Do nothing*/ }
	
	virtual void initMem() override { /*Do nothing*/ }
	virtual void setLearningRates(double alpha, double beta1 = -1., double beta2 = -1.) { /*Do nothing*/ }

	virtual OPTIMIZER_TYPE getType() const override { return OPTIMIZER_TYPE::OPTIMIZER_NONE; }
};

class Optimizer_SGD : public Optimizer {
	//constants={-alpha}
public:
	Optimizer_SGD(TYPE data_type, TYPE computation_type)
	{
		this->data_type = data_type;
		this->computation_type = computation_type;
		
		this->optBuffers = std::vector<TensorDesc*>();
		this->constants  = std::vector<ConstantDesc*>();

		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true));   //-Alpha
	}

	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas) override {
		if (this->data_type != mem->getTypeId())    throw new std::runtime_error("[ERROR] SGD optimizer requires memory to be of the previously specified data type");
		if (this->data_type != deltas->getTypeId()) throw new std::runtime_error("[ERROR] SGD optimizer requires deltas to be of the previously specified data type");
		
		Operation_Copy* op = new Operation_Copy()
			->setInTensor(deltas)
			->setOutTensor(mem)
			->setBlendConstants({
					this->constants[0],
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.)
				});

		graph->addNode(op);
	}

	virtual void initMem() override { /*Do nothing*/ }
	virtual void setLearningRates(double alpha, double beta1 = -1., double beta2 = -1.) override {
		this->constants[0]->setValue(-alpha);
	}

	virtual OPTIMIZER_TYPE getType() const override {
		return OPTIMIZER_TYPE::OPTIMIZER_SGD;
	}
};

class Optimizer_SGD_MOMENT : public Optimizer {
	//constants={-alpha}
public:
	Optimizer_SGD_MOMENT(TYPE data_type, TYPE computation_type)
	{
		this->data_type = data_type;
		this->computation_type = computation_type;

		this->optBuffers = std::vector<TensorDesc*>();
		this->constants = std::vector<ConstantDesc*>();

		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true));   //-Alpha
		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true));   //Beta1
	}

	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas) override {
		if (this->data_type != mem->getTypeId())    throw new std::runtime_error("[ERROR] SGD optimizer requires memory to be of the previously specified data type");
		if (this->data_type != deltas->getTypeId()) throw new std::runtime_error("[ERROR] SGD optimizer requires deltas to be of the previously specified data type");

		Operation_Copy* op = new Operation_Copy()
			->setInTensor(deltas)
			->setOutTensor(mem)
			->setBlendConstants({
					this->constants[0],
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.)
				});

		graph->addNode(op);
	}

	virtual void initMem() override { /*Do nothing*/ }
	virtual void setLearningRates(double alpha, double beta1 = -1., double beta2 = -1.) override {
		this->constants[0]->setValue(-alpha);
	}

	virtual OPTIMIZER_TYPE getType() const override {
		return OPTIMIZER_TYPE::OPTIMIZER_SGD;
	}
};

class Optimizer_Debug : public Optimizer {
	//Basically sgd but safes all the gradients
	
	//constants={-alpha}
public:
	Optimizer_Debug(TYPE data_type, TYPE computation_type)
	{
		this->data_type = data_type;
		this->computation_type = computation_type;

		this->optBuffers = std::vector<TensorDesc*>();
		this->constants  = std::vector<ConstantDesc*>();

		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true));   //-Alpha
	}

	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas) override {
		if (this->data_type != mem->getTypeId())    throw new std::runtime_error("[ERROR] Debug optimizer requires memory to be of the previously specified data type");
		if (this->data_type != deltas->getTypeId()) throw new std::runtime_error("[ERROR] Debug optimizer requires deltas to be of the previously specified data type");

		Operation_Copy* op = new Operation_Copy()
			->setInTensor(deltas)
			->setOutTensor(mem)
			->setBlendConstants({
					this->constants[0],
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.)
				});

		graph->addNode(op);



		TensorDesc* safeGradients = new TensorDesc()
			->setAlignment(deltas->getAlignment())
			->setIsNeeded(true)
			->setSize(deltas->getSize())
			->setStrides(deltas->getStrides())
			->setTypeId(deltas->getTypeId())
			->setUseIndirection(false);
		this->optBuffers.push_back(safeGradients);

		Operation_Copy* op2 = new Operation_Copy()
			->setInTensor(deltas)
			->setOutTensor(safeGradients)
			->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.)
				});
		graph->addNode(op2);
	}

	virtual void initMem() override { /*Do nothing*/ }
	virtual void setLearningRates(double alpha, double beta1 = -1., double beta2 = -1.) override {
		this->constants[0]->setValue(-alpha);
	}

	virtual OPTIMIZER_TYPE getType() const override {
		return OPTIMIZER_TYPE::OPTIMIZER_DBUG;
	}

};

class Optimizer_Adam : public Optimizer {
	//constants={beta1, 1-beta1, beta2, 1-beta2, -alpha * sqrt(1-beta2^t)/(1-beta1^t), epsilon * sqrt(1-beta2^t)}
	double alpha_double, beta1_double, beta2_double, epsilon_double;

public:
	Optimizer_Adam(TYPE data_type_, TYPE computation_type_) :
		alpha_double(-1.), beta1_double(-1.), beta2_double(-1.), epsilon_double(-1.)
	{
		this->data_type        = data_type_;
		this->computation_type = computation_type_;
		
		this->optBuffers = std::vector<TensorDesc*>();
		this->constants  = std::vector<ConstantDesc*>();

		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true)->setValue(0.9   )); //Beta1
		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true)->setValue(0.1   )); //1-beta1
		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true)->setValue(0.999 )); //Beta2
		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true)->setValue(0.001 )); //1-beta2
		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true)                  ); //sqrt(1-beta2^t)/(1-beta1^t)
		this->constants.push_back(new ConstantDesc(this->data_type)->setUseIndirection(true)                  ); //epsilon * sqrt(1-beta2^t)}
	}

	virtual void applyDeltaToMemory(OperationGraph* graph, TensorDesc* mem, TensorDesc* deltas) override {
		if (this->data_type != mem->getTypeId())    throw new std::runtime_error("[ERROR] SGD optimizer requires memory to be of the previously specified data type");
		if (this->data_type != deltas->getTypeId()) throw new std::runtime_error("[ERROR] SGD optimizer requires deltas to be of the previously specified data type");
		
		//Set up optimizer buffers
		TensorDesc* firstOrderMomentum = new TensorDesc()
			->setAlignment(deltas->getAlignment())
			->setIsNeeded(true)
			->setSize(deltas->getSize())
			->setStrides(deltas->getStrides())
			->setTypeId(deltas->getTypeId())
			->setUseIndirection(false);
		TensorDesc* secondOrderMomentum = new TensorDesc()
			->setAlignment(deltas->getAlignment())
			->setIsNeeded(true)
			->setSize(deltas->getSize())
			->setStrides(deltas->getStrides())
			->setTypeId(deltas->getTypeId())
			->setUseIndirection(false);

		this->optBuffers.push_back(firstOrderMomentum);
		this->optBuffers.push_back(secondOrderMomentum);

		//Update optimizer buffers
		Operation_Copy* op_updateBuf1 = new Operation_Copy()
			->setInTensor(deltas)
			->setOutTensor(firstOrderMomentum)
			->setBlendConstants({
					this->constants[1],
					this->constants[0],
				});

		Operation_Pointwise* op_updateBuf2 = new Operation_Pointwise()
			->setInTensor(deltas)
			->setOutTensor(secondOrderMomentum)
			->setBlendConstants({
					this->constants[3],
					this->constants[2],
				})
			->setPointwiseType(POINTWISE_TYPE::POINTWISE_SQUARE);

		//Compute update
		TensorDesc* tmp1 = new TensorDesc()
			->setAlignment(deltas->getAlignment())
			->setIsNeeded(false)
			->setSize(deltas->getSize())
			->setStrides(deltas->getStrides())
			->setTypeId(deltas->getTypeId())
			->setUseIndirection(false);
		TensorDesc* tmp2 = new TensorDesc()
			->setAlignment(deltas->getAlignment())
			->setIsNeeded(false)
			->setSize(deltas->getSize())
			->setStrides(deltas->getStrides())
			->setTypeId(deltas->getTypeId())
			->setUseIndirection(false);

		Operation_Pointwise* opRoot = new Operation_Pointwise()
			->setPointwiseType(POINTWISE_TYPE::POINTWISE_ROOT)
			->setInTensor(firstOrderMomentum)
			->setOutTensor(tmp1)
			->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
				});

		Operation_Pointwise* opAdd = new Operation_Pointwise()
			->setPointwiseType(POINTWISE_TYPE::POINTWISE_ADD)
			->setInTensor(tmp1)
			->setOutTensor(tmp2)
			->setArgument(this->constants[5])
			->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.0)
				});

		//Update
		Operation_Binary* opUpdate = new Operation_Binary()
			->setBinaryType(BINARY_TYPE::BINARY_DIV)
			->setInTensors(secondOrderMomentum, tmp2)
			->setOutTensor(mem)
			->setBlendConstants({
					this->constants[4],
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.0)
				});
						
		//Add all to graph
		graph->addNode(op_updateBuf1);
		graph->addNode(op_updateBuf2);
		graph->addNode(opRoot);
		graph->addNode(opAdd);
		graph->addNode(opUpdate);
	}

	virtual void initMem() override { 
		for (TensorDesc*& buf : this->optBuffers) {
			fillTensorConstant(buf, 0.);
		}
	}
	virtual void setLearningRates(double alpha, double beta1 = -1., double beta2 = -1., uint32_t time_step = (uint32_t)-1) override { 
		if (alpha != -1.) {
			alpha_double = alpha;
		}
		if (beta1 != -1.f) {
			beta1_double = beta1;

			this->constants[0]->setValue(beta1);
			this->constants[1]->setValue(1. - beta1);
		}
		if (beta2 != -1.f) {
			beta2_double = beta2;

			this->constants[2]->setValue(beta2);
			this->constants[3]->setValue(1. - beta2);
		}

		if (time_step != (uint32_t)-1 || alpha != -1. || beta1 != -1.f || beta2 != -1.f) {
			double updateConst  =  -alpha_double * sqrt(1. - pow(beta2_double, time_step)) / (1. - pow(beta1_double, time_step));
			double epsilonConst = epsilon_double * sqrt(1. - pow(beta2_double, time_step));

			this->constants[4]->setValue(updateConst);
			this->constants[5]->setValue(epsilonConst);
		}
	}

	virtual OPTIMIZER_TYPE getType() const override {
		return OPTIMIZER_TYPE::OPTIMIZER_ADAM;
	}
};
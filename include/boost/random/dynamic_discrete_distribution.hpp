#ifndef BOOST_RANDOM_DISCRETE_DISTRIBUTION_HPP_INCLUDED
#define BOOST_RANDOM_DISCRETE_DISTRIBUTION_HPP_INCLUDED
#include <vector>
#include <limits>
#include <istream>
#include <ostream>
#include <queue>
#include <cmath>
#include <string>
#include <bit>
#include <algorithm>
#include <initializer_list>
#include <boost/assert.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/random/detail/config.hpp>
#include <boost/random/detail/operators.hpp>
#include <boost/random/detail/vector_io.hpp>
#include <boost/random/generate_canonical.hpp>



namespace boost { namespace random{

/////////////////////////
// Underlying tree data structure
namespace detail{
template <class IntType = int, class Real = double, std::make_unsigned_t<IntType> fanout = 16>
        class complete_kary_complete_tree {
        public:

        // Compute minimal complete k-ary tree size to store exactly n leaves
        // without needing bounds checks during selection. returns the index of the first leaf and the total number of leaves
        // Returns {total_nodes_excluding_root, leaf_start_index}
        std::pair<IntType, IntType> minimal_tree_shape(std::make_unsigned_t<IntType> n) {

            if (n == 0) return {0, 0}; // no internal nodes, no leaves

            IntType k = boost::core::countr_zero(fanout);
            IntType max_leaf = IntType(1) << (((boost::core::bit_width(n - 1) + k - 1) / k) * k);

            IntType leaf_start = 0; //the index of the first node that can contain leaves
            IntType level = 0; //the index of the first node in the bottom row

            while (true) {
                level = first_child_of(level);
                if (level >= max_leaf) break;
                leaf_start=level;
            }

            IntType total_nodes = leaf_start+n;
            return {total_nodes, leaf_start};
        }



        static_assert((fanout & (fanout - 1)) == 0, "fanout must be power of two");

        using position_type = IntType;

        //constructors
        complete_kary_complete_tree() : data_(1, Real(0)), leaf_start_(0), leaf_count_(0) {}

        explicit complete_kary_complete_tree(IntType leaf_count) {
            resize(leaf_count);
        }

        // Resize to accommodate exactly n leaves
            void resize(IntType n) {
                if (n == 0) {
                    data_.assign(1, Real(0));
                    leaf_start_ = 0;
                    leaf_count_ = 0;
                    return;
                }


                auto [total_nodes, leaf_start] = minimal_tree_shape(n);

                // allocate all internal nodes + leaves (root is kept separately by the derived class)
                data_.resize(total_nodes);
                leaf_count_ = n;
                leaf_start_ = leaf_start;
            }



            IntType size() const {
                return data_.size();
            }

            Real& value_of(position_type p) {
                BOOST_ASSERT(p < data_.size());

                return data_[p];
            }

            const Real& value_of(position_type p) const {
                BOOST_ASSERT(p < data_.size());
                return data_[p];
            }

            // Tree navigation (-1 based indexing). root stored separately
        

            position_type parent_of(position_type i) const {
                BOOST_ASSERT(i > 0);
                return (i >> log2_fanout) - 1;
            }

            position_type first_child_of(position_type i) const {
                return (i + 1) << log2_fanout;
            }

            position_type child_index(position_type parent, IntType child_num) const {
                return first_child_of(parent) + child_num;
            }

            bool is_leaf(position_type i) const {
                return i >= leaf_start_;
            }

            IntType leaf_start() const {
                return leaf_start_;
            }

            IntType leaf_count() const {
                return leaf_count_;
            }

            std::vector<Real,boost::alignment::aligned_allocator<Real, 128> > &data() {return data_;}

            const std::vector<Real,boost::alignment::aligned_allocator<Real, 128> > &data() const {return data_;}

        private:


            std::vector<Real,boost::alignment::aligned_allocator<Real, 128> > data_;
            IntType leaf_start_;
            IntType leaf_count_;
            IntType max_leaf_;

            static constexpr IntType log2_fanout = [] { //This is the lg(fanout) with a base of 2 
                IntType v = fanout;
                IntType r = 0;
                while (v > 1) {
                    v >>= 1;
                    ++r;
                }
                return r;
            }();
        };}

///////////////////////


/**
 * @brief dynamic_discrete_distribution is a highly optimized library for weighted random selection modeled after std::discrete_distribution. It selects an index where the index's probability of being selected is weighted by inputted weights. It satisfies all RandomNumberDistribution requirements except for the requirement of constant time equality comparisons between both distribution objects and parameter objects, which is not possible in discrete distributions.  
 * * This library is designed to be used in discrete event simulation applications in which there is a set of events with different probabilities of occurring, and the simulation must choose an event to occur and update probabilities of other events accordingly. These simulations often involve many more updates than selections, but exact ratios of update to selection differ depending on the application. This library provides efficient update and selection in which the underlying tree data structure can be tuned at compile time to prioritize update over selection to varying degrees. 
 * @tparam IntType is the type of the integers returned by operator(), which is the selection function. IntType must be an unsigned integer type
 * @tparam Real is the type of the weights
 * @tparam Fanout, which must be an positive integer power of 2, controls the branching factor of the underlying complete tree data structure and can be adjusted to change the amount that update is prioritized over selection. update_weight has a runtime of $O(log_{\text{fanout}} N)$ while selection (operator ()) has a runtime of $O( \text{fanout} * (log_{\text{fanout}} N))$ . Additionally, trees with larger fanouts use less memory. 
 * @tparam Precision controls the number of bits of randomness generated during selection. 
 */
template <
    class IntType = int,
    class Real = double,
    IntType Fanout = 16,
    IntType Precision = std::numeric_limits<Real>::digits
>
class dynamic_discrete_distribution {
    static constexpr std::make_unsigned_t<IntType> unsigned_fanout = static_cast<std::make_unsigned_t<IntType>>(Fanout);
    static_assert(Fanout>0 && boost::core::has_single_bit(unsigned_fanout),"template parameter Fanout must be a positive power of 2");
    using This = dynamic_discrete_distribution<IntType, Real, Fanout, Precision>;

public:
    using input_type = Real;  
    using result_type = IntType; 

    /**
     * @brief standard library random number distributions separate state related to the parameters of a distribution and state related to the generation of random numbers by defining a member class param_type that holds the distribution parameters and related data structures. The distribution stores its parameter set in a param_type object. 
     */
    class Param : protected detail::complete_kary_complete_tree<IntType, Real, Fanout> {
        using BaseTree = detail::complete_kary_complete_tree<IntType, Real, Fanout>;
        using PosType = typename BaseTree::position_type;

    public:
        using distribution_type = This;
        
        /**
         * @brief constructs a `param_type` parameter set with a single element 0 with a weight of 1. 
         */
        Param() : Param({1.0}) {}

        /**
         * @brief constructs a `param_type` parameter set with weights equal to the value in `weights` at each index.
         */
        Param(const std::vector<Real>& weights)
            : Param(weights.begin(), weights.end()) {}

        /**
         * @brief constructs a `param_type` parameter set with the weights in the initializer_list `il`. Each weight will have the same index as it did in il. 
         */
        Param(const std::initializer_list<Real>& il)
            : Param(il.begin(), il.end()) {}
  
        /**
         * @brief Constructs a `param_type` parameter set with `count` weights that are generated using function `unary_op`. Each of the weights is equal to ${w_i} = unary_op(xmin + δ(i + 0.5))$, where $δ = (xmax − xmin)count$ and $i ∈ {0, ..., count − 1}$. `xmin` and `xmax` must be such that `δ > 0`. If `count == 0` the effects are the same as of the default constructor.
         */
        template< class UnaryOperation >
        Param( std::size_t count, double xmin, double xmax, UnaryOperation unary_op )
            : Param(weightDistribution(count,xmin,xmax,unary_op)) {}

        /**
         * @brief constructs the distribution from the first and last iterators to a collection of weights of type `weight_type`. Each weight is indexed according to the number of elements between itself and first. 
         */
        template<class InputIt>
        Param(InputIt first, InputIt last)
            : BaseTree()
        {
            IntType n = std::distance(first, last);
            std::make_unsigned_t<IntType> n_unsigned = n;

            //n = 2;

            // Round up leaves to nearest full complete k-ary tree level (power of Fanout)
            if (n <= Fanout){
                max_leaf_ = Fanout;
                num_layers = 1;
            }
            else{
                IntType k = boost::core::countr_zero(unsigned_fanout);
                num_layers = (boost::core::bit_width(n_unsigned - 1) + k - 1) / k;
                max_leaf_ = IntType(1) << (((boost::core::bit_width(n_unsigned - 1) + k - 1) / k) * k);
            }

            leaf_start_ = BaseTree::minimal_tree_shape(n).second;
            BaseTree::resize(n);
            leaf_end_ = n;

            // Copy weights to leaves, pad with zeros
            InputIt it = first;
            for (IntType i = leaf_start_; i < leaf_start_ + n; ++i) {
                weightsum_of(i) = std::max(Real(*it), Real(0));
                ++it;
            }

            // Build sums bottom-up from leaves to root (excluding root)
            for (ptrdiff_t i = leaf_start_ - 1; i >= 0; --i) {
                Real sum = 0;
                PosType first_child = (i + 1) * Fanout; // -1-index adjustment
                for (IntType c = 0; c < Fanout; ++c) {
                    PosType child = first_child + c;
                    if (child >= leaf_start_+n) break;
                    sum += weightsum_of(child);
                }
                weightsum_of(i) = sum;
            }

            // Compute root separately
            Real sum = 0;
            PosType first_child = 0 * Fanout; // root's first child in array
            if (leaf_start_ == 0){
                for (IntType c = 0; c < leaf_end_; ++c) {
                    PosType child = first_child + c;
                    if (child >= max_leaf_) break;
                    sum += weightsum_of(child);
                }
            }
            else{
                for (IntType c = 0; c < Fanout; ++c) {
                    PosType child = first_child + c;
                    if (child >= max_leaf_) break;
                    sum += weightsum_of(child);
                }
            }
            total_weight_ = sum;
        }

        /**
         * @brief returns a vector of probabilities of each integer that could be generated by a dynamic_discrete_distribution using this parameter set. 
         */
        std::vector<Real> probabilities() const {
            std::vector<Real> probs(leaf_end_, Real(0));
            Real total = total_weight();
            if (total <= Real(0)) return probs;

            for (IntType i = 0; i < leaf_end_; ++i) {
                probs[i] = weightsum_of(leaf_start_ + i) / total;
            }
            return probs;
        }

        /**
         * @brief returns the minimum integer that could be generated by a dynamic_discrete_distribution using this parameter set assuming that it is not empty. This minimum will always be 0. 
         */
        static constexpr result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () { return 0; }


        /**
         * @brief returns the maximum integer that could be generated by a dynamic_discrete_distribution using this parameter set assuming that it is not empty. 
         */
        result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return leaf_end_ == 0 ? 0 : leaf_end_ - 1; }

        /**
         * @brief updates weight of int`i`  to `new_weight`. If i is not in this parameter set, the behavior is undefined. 
         */
        void update_weight(IntType i, Real new_weight) {
            BOOST_ASSERT(new_weight >= Real(0));
            BOOST_ASSERT(i <= leaf_end_+leaf_start_);
            BOOST_ASSERT(i>=0);
            i = leaf_start_ + i;
            Real diff = new_weight - weightsum_of(i);
            weightsum_of(i) = new_weight;
            total_weight_ += diff;

            while (i >=Fanout) {                
                i = BaseTree::parent_of(i);
                weightsum_of(i) += diff;
            }
        }

        /**
         * @brief gets the weight of int `i`. If i is not in the parameter set, the behavior is undefined. 
         */
        Real get_weight(IntType i) const {
            BOOST_ASSERT(i <= leaf_end_+leaf_start_);
            BOOST_ASSERT(i>=0);
            return weightsum_of(leaf_start_ + i);
        }

        /**
         * @brief returns the number of elements in this parameter set. 
         */
        IntType size() const { return leaf_end_; }

        /**
         * @brief returns the total of all of the weights in this parameter set.
         */
        Real total_weight() const noexcept {
            if (BaseTree::size() == 0) return Real(0);
            return total_weight_;
        }

        /**
         * @brief adds weights.size() elements to the end of this parameter set with weights given by `weights` vector. 
         */
        void push_back(const std::vector<Real>& weights){
            for (Real w : weights) {
                push_back(w);
            }
        }

        /**
         * @brief adds an element to the end with weight `weight`.
         */
        void push_back(Real weight) {
            expand(leaf_end_+1);   
            update_weight(leaf_end_, weight);
            leaf_end_++;
        }

        /**
         * @brief removes the `count` highest integers. If the parameter set is empty, this function's behavior is undefined. NOTE: in this implementation, this does not deallocate memory or decrease the depth of the underlying tree based data structure. For the purposes of asymptotic analysis, consider "N" to be the largest number of elements ever held in this parameter set. 
         */
        void pop_back(IntType count) {
            for(IntType i = 0; i < count; ++i) {
                pop_back();
            }
        }

        /**
         * @brief removes the highest integer. If the parameter set is empty, this function's behavior is undefined. NOTE: in this implementation, this does not deallocate memory or decrease the depth of the underlying tree based data structure. For the purposes of asymptotic analysis, consider "N" to be the largest number of elements ever held in this parameter set. 
         */
        void pop_back() {
            BOOST_ASSERT(leaf_end_ > 0);
            leaf_end_--;
            update_weight(leaf_end_, Real(0));
        }

        /**
         * @brief compares the weights of each element in `rhs` and `lhs` for equality.
         */
        BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(Param, lhs, rhs){
            IntType sizer = rhs.size();
            if (sizer != lhs.size()){
                return false;
            }
            for (IntType i=0; i<sizer; i++){
                if (rhs.get_weight(i)!=lhs.get_weight(i)){
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief compares the weights of each element in `rhs` and `lhs` for inequality
         */
         BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(Param)

        /**
         * @brief Restores the parameters with data read from `stream`. The formatting flags of `stream` are unchanged. The data must have been written using a stream with the same locale, `CharT` and `Traits` template parameters, otherwise the behavior is undefined. If bad input is encountered, stream.setstate(std::ios::failbit) is called, which may throw std::ios_base::failure. the parameter set is unchanged in that case.
         */
         BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(stream, Param, dist){
           std::vector<Real> newWeights;
           detail::read_vector(stream, newWeights);
           if (stream){
            dist = Param(newWeights);
           }


            return stream;
        }

        /**
         * @brief Writes a textual representation of the parameters to `stream`. The formatting flags and fill character of `stream` are unchanged.
         */
        BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(stream, Param, dist){
            std::vector<Real> leaves(dist.BaseTree::data().begin()+dist.leaf_start_, dist.BaseTree::data().begin()+dist.leaf_start_+dist.leaf_end_ );
            detail::print_vector(stream, leaves);

            return stream;
        }

    private:
        Real& weightsum_of(PosType p) { return BaseTree::value_of(p); }
        const Real& weightsum_of(PosType p) const { return BaseTree::value_of(p); }
        
        // Expand from current leaf_count_ to new_leaf_count (must be larger), structurally, without recomputing
        // new expand: argument is NEW_LEAF_COUNT (number of leaves you want after expansion)
        void expand(IntType new_leaf_count) {
            if (new_leaf_count <= leaf_end_) return; // nothing to do
            auto &data_ = BaseTree::data();
            IntType new_leaf_start = leaf_start_;
            if (new_leaf_count > max_leaf_) {
                max_leaf_ *= Fanout; 
                // old shape
                auto [old_total_nodes, old_leaf_start] = BaseTree::minimal_tree_shape(leaf_end_); 

                // new shape
                auto [new_total_nodes, new_leaf_start] = BaseTree::minimal_tree_shape(new_leaf_count);
                leaf_start_ = new_leaf_start;

                std::vector<IntType> old_starts;
                std::vector<IntType> old_levels;

                std::vector<IntType> new_starts;
                std::vector<IntType> new_levels;

                IntType position = 0;
                IntType sum = 0;

                // Creating the level lengths and level starts for the old size
                while (true) {
                    position = BaseTree::first_child_of(position);
                    old_starts.push_back(sum);
                    IntType level = position - sum;
                    sum += level;
                    old_levels.push_back(level);
                    if (position >= old_total_nodes) break;
                }

                // Creating the level lengths and level starts for the new size
                position = 0;
                sum = 0;
                while (true) {
                    position = BaseTree::first_child_of(position);
                    new_starts.push_back(sum);
                    IntType level = position - sum;
                    sum += level;
                    new_levels.push_back(level);
                    if (position >= new_total_nodes) break;
                }

                // allocate new array (zero-initialized)
                std::vector<Real,boost::alignment::aligned_allocator<Real,128>> new_data(new_total_nodes, Real(0));

                // copy each old level block into the next-deeper level of the new layout.
                // old level i -> new level (i+1). That packs the old block contiguously at the start
                // of the larger new level (the remainder stays zero).
                for (IntType i = 0; i < old_levels.size(); ++i) {
                    IntType old_start = old_starts[i];
                    IntType old_sz    = old_levels[i];
                    IntType new_level_index = i + 1; // destination level index
                    IntType new_start = new_starts[new_level_index];
                    

                    std::copy_n(data_.begin() + old_start, old_sz, new_data.begin() + new_start);
                }

                // recompute only the new top internal layer (level 0) from its children (level 1)
                // top-level nodes occupy global indices new_starts[0] .. new_starts[0]+new_levels[0]-1
                // their first child indices can be computed with first_child_of(parent_index)
                IntType top_count = new_levels[0];
                IntType top_start = new_starts[0];   // usually 0
                for (IntType j = 0; j < top_count; ++j) {
                    IntType parent_idx = top_start + j;
                    // first child in global indexing:
                    IntType first_child = BaseTree::first_child_of(parent_idx);
                    Real sum = Real(0);
                    for (IntType c = 0; c < Fanout; ++c) {
                        IntType child = first_child + c;
                        if (child >= new_total_nodes) break;
                        sum += new_data[child];
                    }
                    new_data[parent_idx] = sum;
                }
                // commit!
                num_layers++;
                data_.swap(new_data);

            }
            else if (new_leaf_count+leaf_start_ >= BaseTree::size()) {
                BaseTree::resize(std::min(new_leaf_count*2,leaf_start_+max_leaf_));
            }
        }

        IntType max_leaf_;
        IntType leaf_end_;   // number of leaves requested by user
        IntType leaf_start_; // index of first leaf in data_
        IntType num_layers;
        Real total_weight_ = 0;

        template< class UnaryOperation >
        static std::vector<Real> weightDistribution(std::size_t count, double xmin, double xmax,UnaryOperation unary_op ){
            std::vector<Real> distro;
            if (count<=0){
                distro.push_back(1);
            }
            else{
                double delta = (xmax - xmin)/count;
                for (size_t i=0; i<count;i++){
                    distro.push_back(unary_op(xmin + delta * (i+0.5)));
                }
            }
            return distro;
        }


        friend class dynamic_discrete_distribution;
    };
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    using param_type = Param;

    /**
     * @brief constructs a distribution with a single element 0 with a weight of 1. 
     */
    dynamic_discrete_distribution() : dynamic_discrete_distribution({1.0}) {}

    /**
     * @brief constructs a distribution with weights equal to the value in `weights` at each index.
     */
    explicit dynamic_discrete_distribution(const std::vector<Real>& weights)
        : dynamic_discrete_distribution(weights.begin(), weights.end()) {}
  
    /**
     * @brief constructs a distribution with the parameters in p. 
     */
    explicit dynamic_discrete_distribution(const Param& p){
        myParam = p;
    }

    /**
     * @brief constructs a distribution with the weights in the initializer_list `il`. Each weight will have the same index as it did in il. 
     */
    dynamic_discrete_distribution(const std::initializer_list<Real>& il)
        : dynamic_discrete_distribution(il.begin(), il.end()) {}
  
    /**
     * @brief Constructs the distribution with `count` weights that are generated using function `unary_op`. Each of the weights is equal to ${w_i} = unary_op(xmin + δ(i + 0.5))$, where $δ = (xmax − xmin)count$ and $i ∈ {0, ..., count − 1}$. `xmin` and `xmax` must be such that `δ > 0`. If `count == 0` the effects are the same as of the default constructor.
     */
    template< class UnaryOperation >
    dynamic_discrete_distribution( std::size_t count, double xmin, double xmax, UnaryOperation unary_op ){
        myParam = Param(count,xmin,xmax,unary_op);
    }

    /**
     * @brief constructs the distribution from the first and last iterators to a collection of weights of type `weight_type`. Each weight is indexed according to the number of elements between itself and first. 
     */
    template<class InputIt>
    dynamic_discrete_distribution(InputIt first, InputIt last){
        myParam = Param(first,last);
    }

    /**
     * @brief Resets the internal state of the distribution object. After a call to this function, the next call to operator() on the distribution object will not be dependent on previous calls to operator(). The distribution still depends on past calls to update_weight, push, and pop. (Note: in this implementation there is no need for reset, so it does no work. It is present for compatibility reasons)
     */
    void reset() {}

    
    /**
     * @brief Generates random numbers distributed according to the weights in the parameter set `param`. If it does not contain at least 1 weight, the behavior is undefined
     */
    template<class URNG>
    IntType operator()(URNG& g,const Param& param) const {
        Real total = param.total_weight();
        assert(total>0 && param.leaf_end_>0);
        if (total <= Real(0)) return 0;
        Real target = boost::random::generate_canonical<Real, Precision, URNG>(g) * total;
        if (target == Real(0)) return 0;
        typename Param::PosType first_child = 0;

        // Start at the top internal node (index 0)
        typename Param::PosType node = 0;

        for(IntType i=0; i<param.num_layers-1;i++){
            Real cumulative = 0;
            bool chosen = false;
            for (IntType c = 0; c < Fanout; ++c) {
                typename Param::PosType child = first_child + c;
                Real w = param.weightsum_of(child);
                if (target < cumulative + w) {
                    node = child;
                    target -=cumulative; 
                    chosen = true;

                    break;
                }
                cumulative += w;
            }
            if (chosen == false){
                node = first_child;
                target -= cumulative;
                target +=param.weightsum_of(first_child);
            }

            first_child = param.BaseTree::first_child_of(node); // first child in array
            
        }
        // last iteration shaved off for extra safety checks due to the possibility of floating point errors
        if (first_child>= param.leaf_start_+param.leaf_end_){ //floating point errors made selection move into empty part of tree at some point
            return this->operator()(g,param); //run selection algorithm again (probability of floating point errors causing this is quite small)
        }
        Real cumulative = 0;
        bool chosen = false;
        for (IntType c = 0; (c < Fanout)&&(c+first_child<param.leaf_end_+param.leaf_start_); ++c) {
            typename Param::PosType child = first_child + c;
            if (child >= param.leaf_end_+param.leaf_start_) break; 
            Real w = param.weightsum_of(child);
            if (target < cumulative + w) {
                node = child;
                target -=cumulative; 
                chosen = true;

                break;
            }
            cumulative += w;
        }
        if (chosen == false){
            node = first_child;
            target -= cumulative;
            target +=param.weightsum_of(first_child);
        }

            // otherwise, node has been updated to chosen_child
        

        // node is now a leaf
        return static_cast<IntType>(node - param.leaf_start_);
    
    }
    /**
     * @brief Generates random numbers that are distributed according to the weights in the distribution's parameter set. If it does not contain at least 1 weight, the behavior is undefined
     */
    template<class URNG>
    result_type operator()(URNG& g) const {
        return this->operator()(g,myParam); 
    }

    /**
     * @brief returns a vector of probabilities of each integer that could be generated by this distribution using its associated parameter set. 
     */
    std::vector<Real> probabilities() const {
        return myParam.probabilities();
    }

    /**
     * @brief gets the distribution's parameter set
     */
    Param param() const {
        return myParam;
    }

    /**
     * @brief sets the distribution's parameter set to `p`
     */
    void param(const Param& p) { 
        myParam = p; 
    }

    /**
     * @brief returns the minimum integer that could be generated by the operator called on the distribution's parameter set assuming that it is not empty. This minimum will always be 0. 
     */
    static constexpr result_type min() { return 0; }

    /**
     * @brief returns the maximum integer that could be generated by the operator called on the distribution's parameter set assuming that it is not empty. 
     */
    result_type max() const { return myParam.max(); }

    /**
     * @brief updates weight of int`i` in the distribution's parameter set to `new_weight`. If i is not in the distribution's parameter set, the behavior is undefined. 
     */
    void update_weight(IntType i, Real new_weight) {
        myParam.update_weight(i, new_weight);
    }

    /**
     * @brief gets the weight of int `i` in the distribution's parameter set. If i is not in the parameter set, the behavior is undefined. 
     */
    Real get_weight(IntType i) const {
        return myParam.get_weight(i);
    }

    /**
     * @brief returns the number of weights in the distribution's parameter set. 
     */
    IntType size() const { return myParam.size(); }

    /**
     * @brief returns the total of all of the weights in the distribution's parameter set.
     */
    Real total_weight() const noexcept {
        return myParam.total_weight();
    }

    /**
     * @brief adds weights.size() elements to the end of the distribution's parameter set with weights given by `weights` vector. 
     */
    void push_back(const std::vector<Real>& weights){
        myParam.push_back(weights);
    }

    /**
     * @brief adds an element to the end of distribution's parameter set with weight `weight`.
     */
    void push_back(Real weight) {
        myParam.push_back(weight);
    }

    /**
     * @brief removes the `count` highest integers from the distribution's parameter set. If the distribution's parameter set is empty, this function's behavior is undefined. NOTE: in this implementation, this does not deallocate memory or decrease the depth of the underlying tree based data structure. For the purposes of asymptotic analysis, consider "N" to be the largest number of elements ever held in the distribution's parameter set. 
     */
    void pop_back(IntType count) {
        myParam.pop_back(count);
    }

    /**
     * @brief removes the highest integer from the distribution's parameter set. If the distribution's parameter set is empty, this function's behavior is undefined. NOTE: in this implementation, this does not deallocate memory or decrease the depth of the underlying tree based data structure. For the purposes of asymptotic analysis, consider "N" to be the largest number of elements ever held in the distribution's parameter set. 
     */
    void pop_back() {
        myParam.pop_back();
    }

    /**
     * @brief compares the parameter sets of `rhs` and `lhs` for equality
     */
    BOOST_RANDOM_DETAIL_EQUALITY_OPERATOR(This, lhs, rhs){
        return (rhs.myParam == lhs.myParam);
    }
    
    /**
     * @brief compares the parameter sets of `rhs` and `lhs` for inequality
     */
    BOOST_RANDOM_DETAIL_INEQUALITY_OPERATOR(This)
    
    
    /**
     * @brief Restores the distribution parameters with data read from `stream`. The formatting flags of `stream` are unchanged. The data must have been written using a stream with the same locale, `CharT` and `Traits` template parameters, otherwise the behavior is undefined. If bad input is encountered, stream.setstate(std::ios::failbit) is called, which may throw std::ios_base::failure. `dist` is unchanged in that case.
     */
    BOOST_RANDOM_DETAIL_ISTREAM_OPERATOR(stream, This, dist){
        return (stream>>dist.myParam);
    }

    /**
     * @brief Writes a textual representation of the distribution parameters to `stream`. The formatting flags and fill character of `stream` are unchanged.
     */
    BOOST_RANDOM_DETAIL_OSTREAM_OPERATOR(stream, This, dist){        
        return (stream<<dist.myParam);
    }

private:
    Param myParam;

};

}}
#include <boost/random/detail/enable_warnings.hpp>

#endif
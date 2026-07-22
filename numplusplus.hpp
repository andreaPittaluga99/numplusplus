#ifndef NUMPLUSPLUS
#define NUMPLUSPLUS

#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <utility>
#include <initializer_list>
#include <cmath>
#include <cstdint>


/*
    Study note:
    it builds a nested std::initializer_list type matching the array rank.

    Example:
        Rank = 3

        initializer_list<
            initializer_list<
                initializer_list<T>
            >
        >
*/

template<typename T, std::size_t Rank>
struct nested_initializer{
    using type = std::initializer_list<typename nested_initializer<T, Rank-1>::type>;
};

/*
    IMPORTANT:
        We need to specify a base case so we dont get to <T, 0>.
*/
template<typename T>
struct nested_initializer<T,1>{
    using type = std::initializer_list<T>;
};


/************************************** Initializer list utilities *********************************************************/

/*
    Converts a nested initializer_list into a flat storage vector.
*/
template<typename T, std::size_t Rank>
void flatten(typename nested_initializer<T, Rank>::type init, std::vector<T>& storage){
    
    /*
        Study note: 
        if constexpr removes the unused branch during compilation.
    */
    if constexpr (Rank == 1){
        for (const auto& value : init){
            storage.push_back(value);
        }
    }
    else{
        for (const auto& sub : init){
            flatten<T, Rank-1>(sub, storage);
        }
    }
}


/*
    Recursively extracts the array shape from a nested initializer_list
    Every level must have the same size, or the initializer list is 
    considered non-rectangular.
*/
template<typename T, std::size_t CurrentRank, std::size_t OriginalRank>
void getShape(typename nested_initializer<T, CurrentRank>::type init, std::array<std::size_t, OriginalRank>& shape, std::size_t actualDepth = 0){
    if(init.size() == 0){
        throw std::invalid_argument( "npp::array error: empty initializer list");
    }
    
    shape[actualDepth] = init.size();

    if constexpr(CurrentRank > 1){
        std::size_t sizeRef = init.begin()->size();
        for (const auto& sub : init){
            if (sub.size() != sizeRef){
                throw std::invalid_argument(
                    "npp::array error: non rectangular initializer list"
                );
            }
        }

        getShape<T, CurrentRank - 1, OriginalRank>(*init.begin(), shape, actualDepth + 1);
    }
}

/*

    (=^..^=)

*/



/************************************* Boolean support *************************************/
class bool_reference {
    std::uint8_t& value;

public:

    bool_reference(std::uint8_t& v)
        : value(v)
    {}

    operator bool() const{
        return value != 0;
    }

    bool_reference& operator=(bool b){
        value = b ? 1 : 0;
        return *this;
    }
};

namespace npp{


    template<typename T, std::size_t Rank>
    class array{
        static_assert(Rank > 0, "npp::array error: Rank must be greater than 0");
        static_assert(std::is_arithmetic_v<T>, "npp::array error: T must be an arithmetic type");

        public:
        /*
            IMPORTANT: we need to specify that we internally use std::vector<std::uint8_t> for booleans.
            This way we can do something like array > 3 and get back a N-tensor of booleans
        */
        using storage_type = std::conditional_t<
            std::is_same_v<T,bool>,
            std::vector<std::uint8_t>,
            std::vector<T>>;

        /***********************************************************************************/
        /********************************** Constructors ***********************************/
        /***********************************************************************************/


        /*
            Creates an uninitialized array with the given shape.

            Storage is allocated according to the total number of elements.
            Elements are value-initialized by std::vector.
        */
        array(const std::array<std::size_t, Rank>& params)
        :   m_shape(params), 
            m_storage(findSize(params)){
                computeStride();
        }

        array(const std::array<std::size_t, Rank>& params, T value)
        :   m_shape(params), 
            m_storage(findSize(params), value){
                computeStride();
        }

        /*
            Constructs an array from a nested initializer_list.

            IMPORTANT: R > 1 is required to avoid constructor ambiguity.
            Without that Rank=1 would also accept initializer_list<T>
            and could create ambiguity with other constructors, for example:

                npp::array<int,1> a({50});
        */
        
        template<std::size_t R = Rank>
        requires (R > 1)
        array(typename nested_initializer<T, R>::type init){   
            flatten<T, R>(init, m_storage);
            getShape<T, R, R>(init, m_shape);
            computeStride();
        }

        array(array&& other) noexcept
        :   m_shape(std::move(other.m_shape)),
            m_storage(std::move(other.m_storage)),
            m_stride(std::move(other.m_stride)){}
        
        array(const array& other)
        :   m_shape(other.m_shape),
            m_storage(other.m_storage),
            m_stride(other.m_stride){}

        /*
            Constructs an array from an existing storage vector.

            The input type is storage_type instead of std::vector<T>
            because bool arrays internally use std::vector<uint8_t>
            instead of std::vector<bool>.
        */
        array(const std::array<size_t,Rank>& shape, const storage_type& data)
        :   m_shape(shape), 
            m_storage(data){
                if(m_storage.size() != findSize(shape)){
                    throw std::invalid_argument("npp::array error: incompatible data size");
                }
                computeStride();
        }

        template<std::size_t N>
        array(const std::array<std::size_t, Rank>& shape, const std::array<T, N>& data)
        :   m_shape(shape), 
            m_storage(data.begin(), data.end()){
                if (N != findSize(shape)){
                    throw std::invalid_argument("npp::array error: incompatible data size");
                }
                computeStride();
        }

        /***********************************************************************************/
        /***************************** Operators overloading *******************************/
        /***********************************************************************************/
        
        template<typename... Index>
        decltype(auto) operator()(Index... indices){
            auto idx = calculateIndex(normalizeIndices(indices...));
        
            if constexpr(std::is_same_v<T,bool>){
                return bool_reference(m_storage[idx]);
            }
            else{
                return (m_storage[idx]);
            }
        }

        template<typename... Index>
        auto operator()(Index... indices) const{
            auto idx = calculateIndex(normalizeIndices(indices...));
            if constexpr(std::is_same_v<T,bool>){
                return m_storage[idx] != 0;
            }
            else{
                return static_cast<const T&>(m_storage[idx]);
            }
        }


        // operator[] is an alias for operator() in the Rank == 1 case
        template<std::size_t R = Rank>
        requires (R == 1)
        decltype(auto) operator[](std::ptrdiff_t i){
            return (*this)(i);
        }

        template<std::size_t R = Rank>
        requires (R == 1)
        auto operator[](std::ptrdiff_t i) const{
            return (*this)(i);
        }

        array& operator= (array&& other) noexcept{
            if(this != &other){
                m_shape = std::move(other.m_shape);
                m_storage = std::move(other.m_storage);
                m_stride = std::move(other.m_stride);
            }
            return *this;
        }

        array& operator=(const array& other){
            if(this != &other){
                m_shape = other.m_shape;
                m_storage = other.m_storage;
                m_stride = other.m_stride;
            }
            return *this;
        }

        [[nodiscard]]
        bool operator==(const array& other) const noexcept {
            return m_shape == other.m_shape && m_storage == other.m_storage;
        }

        [[nodiscard]]
        bool operator!=(const array& other) const noexcept{
            return !(*this == other);
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        array& operator+=(const array& other) {
            return elementwise_assign(other, std::plus<>{});
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        array& operator-=(const array& other) {
            return elementwise_assign(other, std::minus<>{});
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        array& operator*=(const array& other) {
            return elementwise_assign(other, std::multiplies<>{});
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        array& operator/=(const array& other) {
            return elementwise_assign(other, std::divides<>{});
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        [[nodiscard]]
        array operator+ (const array& other) const {
            array res(*this);
            res += other;
            return res; 
        }


        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        [[nodiscard]]
        array operator- (const array& other) const {
            array res(*this);
            res -= other;
            return res; 
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        [[nodiscard]]
        array operator* (const array& other) const {
            array res(*this);
            res *= other;
            return res; 
        }

        template<typename U = T>
        requires (!std::is_same_v<U,bool>)
        [[nodiscard]]
        array operator/ (const array& other) const {
            array res(*this);
            res /= other;
            return res; 
        }


        /*********************************** Scalar operators ******************************/        
        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array& operator+=(Scalar value){
            return scalar_assign(value, std::plus<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array& operator-=(Scalar value){
            return scalar_assign(value, std::minus<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array& operator*=(Scalar value){
            return scalar_assign(value, std::multiplies<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array& operator/=(Scalar value){
            return scalar_assign(value, std::divides<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        [[nodiscard]]
        array operator+ (Scalar value) const{
            array res(*this);
            res += value;
            return res;
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        [[nodiscard]]
        array operator- (Scalar value) const{
            array res(*this);
            res -= value;
            return res;
        }
        
        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        [[nodiscard]]
        array operator* (Scalar value) const{
            array res(*this);
            res *= value;
            return res;
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        [[nodiscard]]
        array operator/ (Scalar value) const{
            array res(*this);
            res /= value;
            return res;
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array<bool, Rank> operator>(Scalar value) const {
            return compare(value, std::greater<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array<bool, Rank> operator<(Scalar value) const {
            return compare(value, std::less<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array<bool, Rank> operator>=(Scalar value) const {
            return compare(value, std::greater_equal<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array<bool, Rank> operator<=(Scalar value) const {
            return compare(value, std::less_equal<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array<bool, Rank> operator==(Scalar value) const {
            return compare(value, std::equal_to<>{});
        }

        template<typename Scalar>
        requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
        array<bool, Rank> operator!=(Scalar value) const {
            return compare(value, std::not_equal_to<>{});
        }
        
        /******************************** Reduction operations ***************************/

        T sum() const {
            T tot = 0;
            for(auto& x : m_storage){
                tot += x;
            }
            return tot;
        }

        T prod() const {
            T tot = 1;
            for(auto& x : m_storage){
                tot *= x;
            }
            return tot;
        }

        T min() const {
            if(m_storage.empty()){
                throw std::invalid_argument("npp::array error: min of empty array");
            }

            return *std::min_element(m_storage.begin(), m_storage.end());
        }

        T max() const {
            if(m_storage.empty()){
                throw std::invalid_argument("npp::array error: max of empty array");
            }

            return *std::max_element(m_storage.begin(), m_storage.end());
        }


        double mean() const{
            if(m_storage.empty()){
                throw std::invalid_argument("npp::array error: mean of empty array");
            }
            double tot = 0;
            for(auto& x : m_storage){
                tot += x;
            }
            return tot/m_storage.size();
        }


        std::size_t argmin_flattened() const {
            if (m_storage.empty()){
                throw std::invalid_argument("npp::array error: argmin of empty array");
            }
            return std::distance(m_storage.begin(), std::min_element(m_storage.begin(), m_storage.end()));
        }

        std::size_t argmax_flattened() const {
            if (m_storage.empty()){
                throw std::invalid_argument("npp::array error: argmin of empty array");
            }
            return std::distance(m_storage.begin(), std::max_element(m_storage.begin(), m_storage.end()));
        }

        std::array<std::size_t, Rank> unravel_index(std::size_t index) const {
            std::array<std::size_t, Rank> coords{};
        
            auto s = stride();
        
            for(std::size_t i = 0; i < Rank; ++i){
                coords[i] = index / s[i];
                index %= s[i];
            }
        
            return coords;
        }

        /******************************** Mathematical functions ***************************/

        array abs() const {
            return unary_transform([](auto x){ return std::abs(x); });
        }

        array sqrt() const {
            return unary_transform([](auto x){ return std::sqrt(x); });
        }

        array exp() const {
            return unary_transform([](auto x){ return std::exp(x); });
        }

        array log() const {
            return unary_transform([](auto x){ return std::log(x); });
        }

        array log10() const {
            return unary_transform([](auto x){ return std::log10(x); });
        }

        array sin() const {
            return unary_transform([](auto x){ return std::sin(x); });
        }

        array cos() const {
            return unary_transform([](auto x){ return std::cos(x); });
        }

        array tan() const {
            return unary_transform([](auto x){ return std::tan(x); });
        }

        /***********************************************************************************/
        /********************************** Iterators **************************************/
        /***********************************************************************************/

        typename storage_type::const_iterator begin() const noexcept { return m_storage.begin(); }

        typename storage_type::iterator begin() noexcept { return m_storage.begin(); }
        
        typename storage_type::const_iterator end() const noexcept{ return m_storage.end(); }

        typename storage_type::iterator end() noexcept { return m_storage.end(); }
        
        typename storage_type::const_iterator cbegin() const noexcept { return m_storage.cbegin(); }

        typename storage_type::const_iterator cend() const noexcept { return m_storage.cend(); }


        /*
            bool arrays are stored as uint8_t instead of using std::vector<bool>:
            vector<bool> uses a proxy reference and cannot provide a valid bool*.
            All other element types return the actual pointer from the underlying storage.
        */
        [[nodiscard]]
        auto data() noexcept {
            if constexpr(std::is_same_v<T,bool>) {
                return static_cast<std::uint8_t*>(nullptr);
            }
            else {
                return m_storage.empty() ? nullptr : m_storage.data();
            }
        }

        [[nodiscard]]
        auto data() const noexcept {
            if constexpr(std::is_same_v<T,bool>) {
                return static_cast<const std::uint8_t*>(nullptr);
            }
            else {
                return m_storage.empty() ? nullptr : m_storage.data();
            }
        }

        /***********************************************************************************/
        /********************************** functions **************************************/
        /***********************************************************************************/

        template<typename... Index>
        decltype(auto) at(Index... indices){
            auto idx = checkIndices(indices...);
        
            if constexpr(std::is_same_v<T,bool>){
                return bool_reference(m_storage[calculateIndex(idx)]);
            }
            else{
                return (m_storage[calculateIndex(idx)]);
            }
        }


        template<typename... Index>
        auto at(Index... indices) const {
            auto idx = checkIndices(indices...);
        
            if constexpr(std::is_same_v<T,bool>){
                return m_storage[calculateIndex(idx)] != 0;
            }
            else{
                return static_cast<const T&>(m_storage[calculateIndex(idx)]);
            }
        }


        void reshape(const std::array<std::size_t, Rank>& params){
            if (params == m_shape){
                return;
            }

            if (findSize(params) != m_storage.size()){
                throw std::invalid_argument("npp::array error: incompatible shape for reshape, reshape would change the number of elements");
            }

            m_shape = params;
            computeStride();
        }

        [[nodiscard]]
        array reshaped(const std::array<std::size_t, Rank>& params) const {
            array ret(*this);
            ret.reshape(params);
            return ret;
        }

        void fill(const T &data){
            std::fill(m_storage.begin(), m_storage.end(), data);
        }

        void swap(array& other) noexcept{
            std::swap(m_shape, other.m_shape);
            std::swap(m_storage, other.m_storage);
            std::swap(m_stride, other.m_stride);
        }

        [[nodiscard]]
        std::size_t rank() const noexcept {return Rank;}
        
        [[nodiscard]]
        std::size_t size() const noexcept {return m_storage.size();}
        
        [[nodiscard]]
        const std::array<std::size_t, Rank>& shape() const noexcept {return m_shape;}
        
        [[nodiscard]]
        const std::array<std::size_t, Rank>& stride() const noexcept {return m_stride;}
        
        [[nodiscard]]
        bool empty() const noexcept { return m_storage.empty();}



    
    private:

        std::array<std::size_t, Rank> m_shape;
        storage_type m_storage;
        std::array<std::size_t, Rank> m_stride;


        /*
            Computes row-major strides.
            Example: 
                shape {2,3,4} ---> stride = {12,4,1}
                
                where 12 is 3*4, 4 is 4*1, etc
        */

        void computeStride(){
            std::size_t mul = 1; 
            for (std::size_t i = Rank; i-- >0;){
                m_stride[i] = mul;
                mul *= m_shape[i]; 
            }
        }



        /*
            Validates indices and converts them to unsigned positions.
            
            Supports negative indexing like Numpy 
            Example:
                -1 refers to the last element of a dimesion
        */

        template<typename... Index>
        std::array<std::size_t, Rank> checkIndices(Index... indices) const{
            static_assert(sizeof...(Index) == Rank, "npp::array error: wrong number of indices");
        
            static_assert((std::is_integral_v<Index> && ...), "npp::array error: indices must be integer types");
        
            std::array<std::common_type_t<Index...>, Rank> idx = {indices...};
            std::array<std::size_t, Rank> result{};
        
            for (std::size_t i = 0; i < Rank; ++i){
                if constexpr (std::is_signed_v<std::common_type_t<Index...>>){
                    if (idx[i] < 0){
                        idx[i] += static_cast<long long>(m_shape[i]);
                    }

                    if (idx[i] < 0 || idx[i] >= static_cast<std::common_type_t<Index...>>(m_shape[i])){
                        throw std::out_of_range("npp::array error: index out of bounds");
                    }
                    
                }
                else{
                    if (idx[i] >= m_shape[i]){
                        throw std::out_of_range("npp::array error: index out of bounds");
                    }
                }
                
                result[i] = static_cast<std::size_t>(idx[i]);
            }
        
            return result;
        }

        template<typename... Index>
        std::array<std::size_t, Rank> normalizeIndices(Index... indices) const {
            static_assert(sizeof...(Index) == Rank);
            static_assert((std::is_integral_v<Index> && ...));
        
            using IndexType = std::common_type_t<Index...>;
        
            std::array<IndexType, Rank> idx{indices...};
            std::array<std::size_t, Rank> result{};
        
            for (std::size_t i = 0; i < Rank; ++i) {
                if constexpr (std::is_signed_v<IndexType>) {
                    if (idx[i] < 0){
                        idx[i] += static_cast<IndexType>(m_shape[i]);
                    }
                }
            
                result[i] = static_cast<std::size_t>(idx[i]);
            }
        
            return result;
        }

        /*
            Finds size for m_storage
            IMPORTANT: 
                We need to check for overflow befor multiplication to prevent
                size_t wrap around
        */

        static std::size_t findSize(const std::array<std::size_t, Rank>& params){
            std::size_t tot = 1;
            for (std::size_t i = 0; i < Rank; ++i){
                if (params[i] != 0 && tot > std::numeric_limits<std::size_t>::max() / params[i]){
                    throw std::overflow_error("npp::array error: size overflow");
                }
                tot *= params[i];
            }
            return tot;
        }

        /*
            Converts N-dimensional coordinates into a linear index
            Example:
                shape  {2,3,4} 
                index  (1,2,3)  
                linear index = 1*12 + 2*4 + 3*1
        */

        std::size_t calculateIndex(const std::array<std::size_t, Rank>& indices) const noexcept{
            std::size_t idx = 0;
            for (std::size_t i = 0; i < Rank; ++i){
                idx += indices[i] * m_stride[i];
            }
            return idx;
        }

        void checkShape(const array& other) const {
            if (m_shape != other.m_shape) {
                throw std::invalid_argument("npp::array error: incompatible array shapes");
            }
        }

        /************************************ Helpers *******************************************/
        template<typename Scalar, typename Op>
        array& scalar_assign(Scalar value, Op op) {
            for (auto& x : m_storage) {
                x = op(x, value);
            }
            return *this;
        }

        template<typename Op>
        array& elementwise_assign(const array& other, Op op) {
            checkShape(other);
        
            for (std::size_t i = 0; i < m_storage.size(); ++i) {
                m_storage[i] = op(m_storage[i], other.m_storage[i]);
            }
        
            return *this;
        }

        template<typename Func>
        array unary_transform(Func f) const {
            array res(*this);
            for(auto& x : res.m_storage)
                x = f(x);
        
            return res;
        }

        template<typename Scalar, typename Compare>
        array<bool, Rank> compare(Scalar value, Compare cmp) const {
            array<bool, Rank> res(m_shape);
            for(std::size_t i = 0; i < m_storage.size(); ++i){
                res[i] = cmp(m_storage[i], value);
            }
        
            return res;
        }

    }; //class

    /************************************ Scalars Operators *********************************************/
    template<typename T, std::size_t Rank, typename Scalar>
    requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
    [[nodiscard]]
    array<T, Rank> operator+(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
    
        for(auto& x : res){
            x += value;
        }
        return res;
    }

    template<typename T, std::size_t Rank, typename Scalar>
    requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
    [[nodiscard]]
    array<T, Rank> operator-(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
    
        for(auto& x : res){
            x = value - x;
        }
        return res;
    }

    template<typename T, std::size_t Rank, typename Scalar>
    requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
    [[nodiscard]]
    array<T, Rank> operator*(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
    
        for(auto& x : res){
            x *= value;
        }
        return res;
    }

    template<typename T, std::size_t Rank, typename Scalar>
    requires (std::is_arithmetic_v<Scalar> && !std::is_same_v<T,bool>)
    [[nodiscard]]
    array<T, Rank> operator/(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
        for(auto& x : res){
            x = value / x;
        }
        return res;
    }
} //namespace npp


#endif
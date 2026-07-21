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

namespace npp{


    template<typename T, std::size_t Rank>
    class array{
        static_assert(Rank > 0, "npp::array error: Rank must be greater than 0");
        static_assert(std::is_arithmetic_v<T>, "npp::array error: T must be an arithmetic type");

        public:

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


        array(const std::array<size_t,Rank>& shape, const std::vector<T>& data)
        :   m_shape(shape), 
            m_storage(std::move(data)){
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
        const T& operator()(Index... indices) const {
            return m_storage[calculateIndex(normalizeIndices(indices...))];
        }

        template<typename... Index>
        T& operator()(Index... indices){
            return m_storage[calculateIndex(normalizeIndices(indices...))];
        }


        // operator[] is an alias for operator() in the Rank == 1 case
        template<std::size_t R = Rank>
        requires (R == 1)
        T& operator[](std::ptrdiff_t i){
            return (*this)(i);
        }

        template<std::size_t R = Rank>
        requires (R == 1)
        const T& operator[](std::ptrdiff_t i) const{
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

        array& operator+= (const array& other){
            checkShape(other);
            for (std::size_t i = 0; i < m_storage.size(); ++i){
                m_storage[i] += other.m_storage[i];
            }
            return *this;
        }

        [[nodiscard]]
        array operator+ (const array& other) const {
            array res(*this);
            res += other;
            return res; 
        }

        array& operator-= (const array& other){
            checkShape(other);
            for (std::size_t i = 0; i < m_storage.size(); ++i){
                m_storage[i] -= other.m_storage[i];
            }
            return *this;
        }

        [[nodiscard]]
        array operator- (const array& other) const {
            array res(*this);
            res -= other;
            return res; 
        }

        array& operator*= (const array& other){
            checkShape(other);
            for (std::size_t i = 0; i < m_storage.size(); ++i){
                m_storage[i] *= other.m_storage[i];
            }
            return *this;
        }

        [[nodiscard]]
        array operator* (const array& other) const {
            array res(*this);
            res *= other;
            return res; 
        }

        array& operator/= (const array& other){
            checkShape(other);
            for (std::size_t i = 0; i < m_storage.size(); ++i){
                m_storage[i] /= other.m_storage[i];
            }
            return *this;
        }

        [[nodiscard]]
        array operator/ (const array& other) const {
            array res(*this);
            res /= other;
            return res; 
        }


        /*********************************** Scalar operators ******************************/

        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        array& operator+= (Scalar value){
            for(auto& x : m_storage){
                x += value;
            }
            return *this;
        }


        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        [[nodiscard]]
        array operator+ (Scalar value) const{
            array res(*this);
            res += value;
            return res;
        }
        
        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        array& operator-= (Scalar value){
            for(auto& x : m_storage){
                x -= value;
            }
            return *this;
        }


        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        [[nodiscard]]
        array operator- (Scalar value) const{
            array res(*this);
            res -= value;
            return res;
        }
        
        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        array& operator*= (Scalar value){
            for(auto& x : m_storage){
                x *= value;
            }
            return *this;
        }


        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        [[nodiscard]]
        array operator* (Scalar value) const{
            array res(*this);
            res *= value;
            return res;
        }
        
        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        array& operator/= (Scalar value){
            for(auto& x : m_storage){
                x /= value;
            }
            return *this;
        }


        template<typename Scalar>
        requires std::is_arithmetic_v<Scalar>
        [[nodiscard]]
        array operator/ (Scalar value) const{
            array res(*this);
            res /= value;
            return res;
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
        /***********************************************************************************/
        /********************************** Iterators **************************************/
        /***********************************************************************************/

        typename std::vector<T>::const_iterator begin() const noexcept { return m_storage.begin(); }
        
        typename std::vector<T>::iterator begin() noexcept { return m_storage.begin(); }
        
        typename std::vector<T>::const_iterator end() const noexcept{ return m_storage.end(); }

        typename std::vector<T>::iterator end() noexcept { return m_storage.end(); }
        
        typename std::vector<T>::const_iterator cbegin() const noexcept { return m_storage.cbegin(); }

        typename std::vector<T>::const_iterator cend() const noexcept { return m_storage.cend(); }

        [[nodiscard]]
        const T* data() const noexcept { return m_storage.empty() ? nullptr : m_storage.data(); }

        [[nodiscard]]
        T* data() noexcept { return m_storage.empty() ? nullptr : m_storage.data(); }

        /***********************************************************************************/
        /********************************** functions **************************************/
        /***********************************************************************************/

        template<typename... Index>
        const T& at(Index... indices) const {
            auto idx = checkIndices(indices...);
            return m_storage[calculateIndex(idx)];
        }

        template<typename... Index>
        T& at(Index... indices){
            auto idx = checkIndices(indices...);
            return m_storage[calculateIndex(idx)];
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
        std::vector<T> m_storage;
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

    }; //class

    /************************************ Scalars Operators *********************************************/
    template<typename T, std::size_t Rank, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
    [[nodiscard]]
    array<T, Rank> operator+(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
    
        for(auto& x : res){
            x += value;
        }
        return res;
    }

    template<typename T, std::size_t Rank, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
    [[nodiscard]]
    array<T, Rank> operator-(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
    
        for(auto& x : res){
            x = value - x;
        }
        return res;
    }

    template<typename T, std::size_t Rank, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
    [[nodiscard]]
    array<T, Rank> operator*(Scalar value, const array<T, Rank>& arr){
        array<T, Rank> res(arr);
    
        for(auto& x : res){
            x *= value;
        }
        return res;
    }

    template<typename T, std::size_t Rank, typename Scalar>
    requires std::is_arithmetic_v<Scalar>
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
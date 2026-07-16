#ifndef NUMPLUSPLUS
#define NUMPLUSPLUS

#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <limits>


namespace npp{
    template<typename T, std::size_t Rank>
    class array{
        static_assert(Rank > 0, "npp::array error: Rank must be greater than 0");

        public:
        /***********************************************************************************/
        /********************************** Constructors ***********************************/
        /***********************************************************************************/
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

        array(array&& other) noexcept
        :   m_shape(std::move(other.m_shape)),
            m_storage(std::move(other.m_storage)),
            m_stride(std::move(other.m_stride)){}
        
        array(const array& other)
        :   m_shape(other.m_shape),
            m_storage(other.m_storage),
            m_stride(other.m_stride){}
        /***********************************************************************************/
        /***************************** Operators overloading *******************************/
        /***********************************************************************************/
        
        template<typename... Index>
        const T& operator()(Index... indices) const {
            auto idx = checkIndices(indices...);
            return m_storage[calculateIndex(idx)];
        }

        template<typename... Index>
        T& operator()(Index... indices){
            auto idx = checkIndices(indices...);
            return m_storage[calculateIndex(idx)];
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

        /***********************************************************************************/
        /********************************** Iterators **************************************/
        /***********************************************************************************/


        typename std::vector<T>::const_iterator begin() const { return m_storage.begin(); }
        typename std::vector<T>::iterator begin(){ return m_storage.begin(); }
        
        typename std::vector<T>::const_iterator end() const{ return m_storage.end(); }

        typename std::vector<T>::iterator end(){ return m_storage.end(); }
        typename std::vector<T>::const_iterator cbegin() const { return m_storage.cbegin(); }

        typename std::vector<T>::const_iterator cend() const { return m_storage.cend(); }

        const T* data() const { return m_storage.empty() ? nullptr : m_storage.data(); }

        T* data() { return m_storage.empty() ? nullptr : m_storage.data(); }

        /***********************************************************************************/
        /********************************** functions **************************************/
        /***********************************************************************************/

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

        void fill(const T &data){
            std::fill(m_storage.begin(), m_storage.end(), data);
        }

        void swap(array& other) noexcept{
            std::swap(m_shape, other.m_shape);
            std::swap(m_storage, other.m_storage);
            std::swap(m_stride, other.m_stride);
        }

        std::size_t rank() const {return Rank;}
        std::size_t size() const {return m_storage.size();}
        const std::array<std::size_t, Rank>& shape() const {return m_shape;}
        const std::array<std::size_t, Rank>& stride() const {return m_stride;}
        bool empty() const { return m_storage.empty();}



    
    private:

        std::array<std::size_t, Rank> m_shape;
        std::vector<T> m_storage;
        std::array<std::size_t, Rank> m_stride;

        void computeStride(){
            std::size_t mul = 1; 
            for (std::size_t i = Rank; i-- >0;){
                m_stride[i] = mul;
                mul *= m_shape[i]; 
            }
        }

        template<typename... Index>
        std::array<std::size_t, Rank> checkIndices(Index... indices) const{
            static_assert(sizeof...(Index) == Rank, "npp::array error: wrong number of indices");
        
            static_assert((std::is_integral_v<Index> && ...), "npp::array error: indices must be integer types");
        
            std::array<std::common_type_t<Index...>, Rank> idx = {indices...};
            std::array<std::size_t, Rank> result{};
        
            for (std::size_t i = 0; i < Rank; ++i){
                //negative index like numpy
                if constexpr (std::is_signed_v<std::common_type_t<Index...>>){
                    if (idx[i] < 0){
                        idx[i] += static_cast<long long>(m_shape[i]);
                    }
                    //check out of bounds
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

        

        std::size_t calculateIndex(const std::array<std::size_t, Rank>& indices) const {
            std::size_t idx = 0;
            for (std::size_t i = 0; i < Rank; ++i){
                idx += indices[i] * m_stride[i];
            }
            return idx;
        }
    }; //class
} //namespace npp


#endif
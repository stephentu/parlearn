#pragma once

#include <cassert>
#include <amd64.hh>

typedef uint64_t version_t;

/**
 * Transactional vector - wraps a (non-sparse) vector and provides quick
 * transactions over the vector (that the size must remain constant is the only
 * restriction)
 */
template <typename T>
class standard_tvec {
  friend class txn;
public:
  standard_tvec(size_t n)
    : impl_(n) {}

  static const version_t LOCK_MASK = 0x1;

  /**
   * Does not have read-own-write semantics
   */
  class txn {
  public:
    typedef std::pair<size_t, version_t> read_t;
    typedef std::pair<size_t, T> write_t;

    txn(standard_tvec<T> *timpl)
      : timpl_(timpl) {}

    inline T
    read(size_t idx)
    {
    retry:
      const version_t v = timpl_->stablev(idx);
      const T ret = timpl_->unsaferead(idx);
      if (unlikely(!timpl_->checkv(idx, v))) {
        nop_pause();
        goto retry;
      }
      reads_.emplace_back(idx, v);
      return ret;
    }

    inline void
    write(size_t idx, const T &t)
    {
      writes_.emplace_back(idx, t);
    }

    inline bool
    commit()
    {
      for (auto &p : reads_) {
        if (unlikely(timpl_->unstablev(p.first) != p.second)) {
          clear();
          return false;
        }
      }
      // XXX(stephentu): check if std::sort() is stable!
      std::sort(writes_.begin(), writes_.end(),
          [](const write_t &a, const write_t &b) { return a.first < b.first; });
      for (auto &p : writes_) {
        timpl_->lock(p.first);
        timpl_->unsafewrite(p.first, p.second);
        timpl_->unlock(p.first);
      }
    }

  private:
    inline void
    clear()
    {
      reads_.clear();
      writes_.clear();
    }

    standard_tvec<T> *timpl_;
    std::vector< read_t > reads_;
    std::vector< write_t > writes_;
  };

  inline version_t
  stablev(size_t idx) const
  {
    assert(idx < impl_.size());
    const auto &ref = impl_[idx];
    version_t ret = ref.first;
    while (ret & LOCK_MASK) {
      nop_pause();
      ret = ref.first;
    }
    assert(!(ret & LOCK_MASK));
    compiler_barrier();
    return ret;
  }

  inline version_t
  unstablev(size_t idx) const
  {
    assert(idx < impl_.size());
    return impl_[idx].first;
  }

  inline bool
  checkv(size_t idx, version_t v) const
  {
    assert(!(v & LOCK_MASK));
    assert(idx < impl_.size());
    compiler_barrier();
    return impl_[idx].first == v;
  }

  inline T
  unsaferead(size_t idx) const
  {
    assert(idx < impl_.size());
    return impl_[idx].second;
  }

  inline void
  unsafewrite(size_t idx, const T &t)
  {
    assert(idx < impl_.size());
    //assert(impl_[idx].first & LOCK_MASK);
    impl_[idx].second = t;
  }

  inline void
  lock(size_t idx)
  {
    assert(idx < impl_.size());
    version_t v = impl_[idx].first;
    while ((v & LOCK_MASK) ||
           !__sync_bool_compare_and_swap(&impl_[idx].first, v, v | LOCK_MASK)) {
      nop_pause();
      v = impl_[idx].first;
    }
    compiler_barrier();
  }

  inline void
  unlock(size_t idx)
  {
    compiler_barrier();
    assert(idx < impl_.size());
    const version_t v = impl_[idx].first;
    assert(v & LOCK_MASK);
    const version_t newv = (((v>>1)+1)<<1);
    assert(!(v & LOCK_MASK));
    impl_[idx].first = newv;
  }

	inline void
	unsafesnapshot(standard_vec<T> &v) const
	{
		v.resize(impl_.size());
		for (size_t i = 0; i < impl_.size(); i++)
			v[i] = impl_[i].second;
	}

private:
  typedef std::pair<version_t, T> entry_t;
  std::vector< entry_t > impl_;
};

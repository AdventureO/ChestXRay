import urllib
from tqdm import tqdm
import os


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


url = 'https://public.boxcloud.com/d/1/i651uXuvZV6dpxzULNaiz4vvM8W7VJapdmJ58tejwYQS81V3ezm5y48V_3k8e6ZC4VbyzkBgLhSmwajjEi0Ncmqlet8KaWalLBUR8lgdmCL5t-KiH-vIpB4vpCFKS_U6YoZJaxWX6gicF9jz0BM6a-ORfz8JGvPG-b4zWF6wzh-DgD9WKbLma8lrS7WPbEFJQzD2wS6shG4LtkBN5f2BUiAJl3PQl3PYW0aJNf6wGTR0Ce1jG_Mz25jboF7gGR4Ra8xqMb_66V6uTEJlcxLPnkkyjcaFtlBO0ArH4n9ybBcCNwaJHRFYvH_zkv4sRy0c2h57OTsIoQ6H0kkXUWlEK4EBTpJJCXzr2YRS8b6szJifIYEgqcYFoThwPs3D2sRzJcP1Zr9svacUi806ml1Ge8DQ2S7hxYVMW-irN8MexCzQtXyPIV_zReyqd2wTkRUajMiqScfOeW7NPH5zswOqCDOeJ5tf45yujMbeqfBZLVkYo1enSJ_JHtL_noXZTcIdo37w2gAhgKGjIo6MHbGjHuR2GbvpxC7zoJc_Jtc5GkQe0p_pQuTd2JUebzLf0x_g-oOacF0Ih4RwVxBC0tW_pT2gpY-u8pDhfD1h-ungrcxALvTTE7_jHOwcHaC9JfnWb26jZ911CryaIQJOQ06HkOOMKBlv-j86z25M8chf3ZikQlAY0YyawzYZZCmLytzE1i19YJpFwGWGgo1AR_mkKw32xxRYNL4Llo1g-0P7fmqFEgdw9d0cXOk4cUEXODRagjtQh8XkiD5mWGfoqgPsiafg6cqDHxjt01GQVU_T2vAOD_Fsg3lsZ8OIb0FjqrI9DG7au7useR4ERd4LKR5WoELfhc8-Y7e1e2TQrF2B1INUNiwL_xhpilw2FgJ8xfi0VrhXJs149Xale0y6rS7D05UcVRz4_YcAX2LTdU0f-zKYAPy_qLV4HYiercGKMkzIp7TLpEgiJJbXkZ6TdwONV5n9vpXfQmMiOpdV60e_mCrc7yGuRuqlO0JmFPlbNtStbhX2u8EHvafOsq9P0s-Q_FozHG29W0yOiEMGe9FuccURmc59NXCdSHoNf9HjNVnPeZTgSO3aFOmcI0Qz7VCuJg41BUTiIBCFO7QpKjPaCs9OTjy9UMLODv5-Q7TprVOPZlYKPmkszV1E/download'


with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
              desc=url.split('/')[-1]) as t:  # all optional kwargs
	urllib.urlretrieve(url, 
		       filename='dataset/images_001.tar.gz',
                       reporthook=t.update_to, 
		       data=None)

